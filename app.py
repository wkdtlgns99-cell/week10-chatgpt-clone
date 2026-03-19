import os
import json
import time
import uuid
import datetime
import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

CHATS_DIR = "chats"
MEMORY_FILE = "memory.json"


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def safe_load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dirs():
    os.makedirs(CHATS_DIR, exist_ok=True)
    if not os.path.exists(MEMORY_FILE):
        safe_write_json(MEMORY_FILE, {})


def chat_path(chat_id):
    return os.path.join(CHATS_DIR, f"{chat_id}.json")


def load_all_chats():
    chats = []
    if not os.path.isdir(CHATS_DIR):
        return chats
    for fn in os.listdir(CHATS_DIR):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(CHATS_DIR, fn)
        obj = safe_load_json(p, None)
        if not isinstance(obj, dict):
            continue
        if "id" not in obj or "messages" not in obj:
            continue
        chats.append(obj)
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats


def new_chat():
    cid = str(uuid.uuid4())
    t = now_iso()
    obj = {"id": cid, "title": "New Chat", "created_at": t, "updated_at": t, "messages": []}
    safe_write_json(chat_path(cid), obj)
    return obj


def save_chat(obj):
    obj["updated_at"] = now_iso()
    safe_write_json(chat_path(obj["id"]), obj)


def delete_chat(cid):
    try:
        os.remove(chat_path(cid))
    except Exception:
        pass


def get_token():
    try:
        v = st.secrets.get("HF_TOKEN", "")
        return v.strip() if isinstance(v, str) else ""
    except Exception:
        return ""


def hf_headers(token):
    return {"Authorization": f"Bearer {token}"}


def build_system_prompt(memory_obj):
    if not memory_obj:
        return "You are a helpful assistant."
    return "You are a helpful assistant. User memory (JSON): " + json.dumps(memory_obj, ensure_ascii=False)


def hf_chat(token, messages, stream=False, timeout=60):
    payload = {"model": HF_MODEL, "messages": messages, "max_tokens": 512, "stream": bool(stream)}
    r = requests.post(HF_ENDPOINT, headers=hf_headers(token), json=payload, stream=stream, timeout=timeout)
    return r


def parse_chat_completion_json(resp_json):
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return ""


def sse_text_chunks(resp):
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                yield delta
        except Exception:
            continue


def extract_memory(token, user_text):
    prompt = (
        "Given the user's message, extract personal facts or preferences as a JSON object. "
        "Only include stable, non-sensitive preferences (e.g., preferred language, hobbies, favorite topics). "
        "If none, return {}. User message:\n"
        f"{user_text}"
    )
    msgs = [{"role": "system", "content": "Return ONLY valid JSON."}, {"role": "user", "content": prompt}]
    try:
        r = hf_chat(token, msgs, stream=False, timeout=45)
        if r.status_code != 200:
            return {}
        out = parse_chat_completion_json(r.json())
        obj = json.loads(out)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def merge_memory(old, new):
    if not isinstance(old, dict):
        old = {}
    if not isinstance(new, dict):
        return old
    for k, v in new.items():
        old[k] = v
    return old


def main():
    ensure_dirs()
    st.set_page_config(page_title="My AI Chat", layout="wide")

    token = get_token()
    mem = safe_load_json(MEMORY_FILE, {})
    if not isinstance(mem, dict):
        mem = {}

    if "chats" not in st.session_state:
        st.session_state["chats"] = load_all_chats()
    if "active_chat_id" not in st.session_state:
        st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"] if st.session_state["chats"] else ""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.header("Chats")
        if st.button("New Chat"):
            obj = new_chat()
            st.session_state["chats"] = load_all_chats()
            st.session_state["active_chat_id"] = obj["id"]
            st.session_state["messages"] = []
            st.rerun()

        with st.expander("User Memory", expanded=True):
            st.json(mem)
            if st.button("Clear Memory"):
                mem = {}
                safe_write_json(MEMORY_FILE, mem)
                st.rerun()

        st.divider()

        chats = st.session_state["chats"]
        active = st.session_state["active_chat_id"]

        if not chats:
            st.caption("No chats yet. Click 'New Chat'.")
        else:
            for c in chats:
                cid = c["id"]
                cols = st.columns([0.82, 0.18])
                label = c.get("title", "Chat")
                ts = c.get("updated_at", "")[:16].replace("T", " ")
                btn_label = f"{label} — {ts}" if ts else label
                if cols[0].button(btn_label, key=f"open_{cid}", use_container_width=True):
                    st.session_state["active_chat_id"] = cid
                    obj = safe_load_json(chat_path(cid), c)
                    st.session_state["messages"] = obj.get("messages", [])
                    st.rerun()
                if cols[1].button("✕", key=f"del_{cid}"):
                    delete_chat(cid)
                    st.session_state["chats"] = load_all_chats()
                    if st.session_state["active_chat_id"] == cid:
                        st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"] if st.session_state["chats"] else ""
                        st.session_state["messages"] = []
                    st.rerun()

    st.title("My AI Chat")

    if not token:
        st.error('Missing Hugging Face token. Create ".streamlit/secrets.toml" with HF_TOKEN="...".')
        st.stop()

    active = st.session_state["active_chat_id"]
    if active and not st.session_state["messages"]:
        obj = safe_load_json(chat_path(active), None)
        if isinstance(obj, dict):
            st.session_state["messages"] = obj.get("messages", [])

    for m in st.session_state["messages"]:
        with st.chat_message(m.get("role", "assistant")):
            st.write(m.get("content", ""))

    user_text = st.chat_input("Type a message and press Enter")
    if not user_text:
        return

    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    if not active:
        obj = new_chat()
        st.session_state["active_chat_id"] = obj["id"]
        st.session_state["chats"] = load_all_chats()
        active = obj["id"]

    obj = safe_load_json(chat_path(active), {"id": active, "messages": []})
    if obj.get("title") in ["New Chat", "", None]:
        obj["title"] = user_text[:24] + ("..." if len(user_text) > 24 else "")

    api_msgs = [{"role": "system", "content": build_system_prompt(mem)}] + st.session_state["messages"]

    with st.chat_message("assistant"):
        ph = st.empty()
        acc = ""
        try:
            resp = hf_chat(token, api_msgs, stream=True, timeout=120)
            if resp.status_code != 200:
                ph.error(f"API error: {resp.status_code} {resp.text[:200]}")
                return
            for ch in sse_text_chunks(resp):
                acc += ch
                ph.write(acc)
                time.sleep(0.02)
        except requests.exceptions.RequestException as e:
            ph.error(f"Network/API error: {e}")
            return

    st.session_state["messages"].append({"role": "assistant", "content": acc})

    mem_new = extract_memory(token, user_text)
    mem = merge_memory(mem, mem_new)
    safe_write_json(MEMORY_FILE, mem)

    obj["messages"] = st.session_state["messages"]
    save_chat(obj)
    st.session_state["chats"] = load_all_chats()

    st.rerun()


if __name__ == "__main__":
    main()
