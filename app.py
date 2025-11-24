import os
import random
import string
import re
import json
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, g, make_response
from dotenv import load_dotenv
from datetime import datetime, timezone
from functools import wraps
import requests

# --- AI Imports ---
import google.generativeai as genai
import stability_sdk.client as StabilityClient
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# --- Firebase & Cloudinary ---
import firebase_admin
from firebase_admin import credentials, firestore, auth
import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- VERCEL FIX 1: Load Environment Variables ---
load_dotenv() 

# --- VERCEL FIX 2: Handle serviceAccountKey.json ---
service_account_json_string = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

try:
    if service_account_json_string:
        service_account_info = json.loads(service_account_json_string)
        cred = credentials.Certificate(service_account_info)
    else:
        cred = credentials.Certificate('serviceAccountKey.json')
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase Init Error (Non-fatal if mostly generic): {e}")

db = firestore.client()

# --- VERCEL FIX 3: Flask Config ---
base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_strong_default_secret_key_12345')

# --- API Configurations ---
api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=api_key) 
model_flash = genai.GenerativeModel('gemini-2.5-flash') 

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PRO_MODEL_NAME = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

stability_api_key = os.environ.get("STABILITY_API_KEY")
if stability_api_key:
    stability_api = StabilityClient.StabilityInference(
        key=stability_api_key, verbose=True, engine="stable-diffusion-xl-1024-v1-0"
    )

cloudinary.config( 
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.environ.get("CLOUDINARY_API_KEY"), 
    api_secret = os.environ.get("CLOUDINARY_API_SECRET") 
)

# === Authentication Decorator ===
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = request.headers.get('Authorization')
        if not id_token:
            id_token = session.get('id_token')
            if not id_token:
                return redirect(url_for('login'))

        if id_token.startswith('Bearer '):
            id_token = id_token.split(' ')[1]
            
        try:
            decoded_token = auth.verify_id_token(id_token)
            g.user = decoded_token
            g.user_uid = decoded_token['uid']
        except Exception as e:
            print(f"Token verification error: {e}")
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

# --- Helper Functions ---
def generate_room_code(length=4):
    while True:
        code = ''.join(random.choices(string.digits, k=length))
        room_ref = db.collection('rooms').document(code)
        if not room_ref.get().exists:
            return code

def build_gemini_prompt(room_data, user_nickname, message_text):
    prompt_lines = [
        f"You are {room_data['bot_name']}. Your personality is: {room_data['bot_personality']}.",
        f"Your appearance is: {room_data.get('bot_appearance', 'not specified')}",
        "You are in a role-playing game where users are trying to win your affection.",
        "\nCurrent affection levels:"
    ]
    if not room_data.get('users'):
        prompt_lines.append("No one is in the room yet.")
    else:
        for user_id, data in room_data['users'].items():
            prompt_lines.append(f"- {data['nickname']}: {data['score']}%")

    prompt_lines.append(f"\nThe current scenario: {room_data['start_scenario']}")
    prompt_lines.append("\nHere is the recent chat history (max 10):")
    
    sorted_messages = sorted(room_data.get('messages', []), key=lambda m: m.get('timestamp'))
    for msg in sorted_messages[-10:]:
        sender = msg['user_id']
        if sender == room_data['bot_name']: sender_display = "You"
        elif sender == "System": sender_display = "System"
        else: sender_display = next((data['nickname'] for uid, data in room_data.get('users', {}).items() if uid == sender), sender)
        prompt_lines.append(f"{sender_display}: {msg['text']}")

    prompt_lines.append(f"\n--- NEW MESSAGE ---")
    prompt_lines.append(f"{user_nickname}: {message_text}")
    prompt_lines.append("\n--- YOUR TASK ---")
    # === CHANGED SECTION BELOW ===
    prompt_lines.append(
        "1. Respond in character to the new message.\n"
        f"2. Evaluate affection change for {user_nickname} ONLY (-20 to 20).\n"
        f"Difficulty: {room_data['difficulty']}/10.\n"
        "IMPORTANT: Return ONLY valid JSON. Do not use Markdown formatting.\n"
        "Format: {\"response\": \"your text response\", \"affection_change\": integer}"
    )
    return "\n".join(prompt_lines)

def build_openrouter_prompt_messages(room_data, user_nickname, message_text):
    system_prompt = (
        f"You are {room_data['bot_name']}. "
        f"Personality: {room_data['bot_personality']}. "
        f"Chatting privately with the winner, {user_nickname}."
    )
    messages = [{"role": "system", "content": system_prompt}]
    
    sorted_messages = sorted(room_data.get('messages', []), key=lambda m: m.get('timestamp'))
    for msg in sorted_messages[-15:]:
        if msg['user_id'] == room_data['bot_name']:
            messages.append({"role": "assistant", "content": msg['text']})
        elif msg['user_id'] != 'System':
            messages.append({"role": "user", "content": msg['text']})
    
    messages.append({"role": "user", "content": message_text})
    return messages

def parse_gemini_response(text):
    try:
        # 1. Clean up Markdown code blocks if Gemini adds them
        text = text.replace("```json", "").replace("```", "").strip()

        # 2. Find the JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match: 
            return {"response": text, "affection_change": 0}
        
        data = json.loads(match.group(0))
        
        # 3. Safely extraction affection change
        aff_change = data.get("affection_change", 0)
        
        # If the AI still messes up and returns a dict/list, force it to 0 to prevent crash
        if isinstance(aff_change, (dict, list, str)):
            try:
                # Try to convert string to int
                aff_change = int(aff_change)
            except:
                # If it's a dict/list/invalid string, just ignore the score change
                print(f"AI returned invalid affection type: {type(aff_change)}")
                aff_change = 0

        return {"response": data.get("response", "..."), "affection_change": int(aff_change)}
    except Exception as e:
        print(f"JSON Parse Error: {e} | Raw Text: {text}")
        # Return the raw text cleaned of json tags so the user at least sees the message
        clean_text = text.replace('{"response":', '').replace('}', '').replace('"', '')
        return {"response": clean_text, "affection_change": 0}

# --- Routes ---

@app.route("/")
def index():
    return redirect(url_for('public_bots'))

@app.route("/login")
def login(): return render_template("login.html")

@app.route("/register")
def register(): return render_template("register.html")

@app.route("/set-token", methods=["POST"])
def set_token():
    data = request.get_json()
    session['id_token'] = data['token']
    return jsonify({"success": True})

@app.route("/logout")
def logout():
    session.pop('id_token', None)
    flash("Logged out.")
    return redirect(url_for('login'))

@app.route("/public-bots")
def public_bots():
    bots = []
    active_sessions = {} # Map: parent_bot_id -> instance_room_code

    # 1. Fetch Public Bots (Templates ONLY)
    try:
        # Fetch more docs to account for the filtering we are about to do
        bots_ref = db.collection('rooms').where('is_public', '==', True).limit(100)
        
        for doc in bots_ref.stream():
            bot_data = doc.to_dict()
            
            # === THE FIX: Filter out Hosted Instances ===
            # If 'is_hosted' is True, it's a game session, not a public bot template. Skip it.
            if bot_data.get('is_hosted') == True:
                continue
            # ============================================

            bot_data['id'] = doc.id
            bots.append(bot_data)
    except Exception as e:
        print(f"Error fetching bots: {e}")

    # 2. If User is Logged In, Fetch their Active Instances (Unchanged)
    id_token = session.get('id_token')
    if id_token:
        try:
            decoded_token = auth.verify_id_token(id_token)
            user_uid = decoded_token['uid']
            
            my_instances = db.collection('rooms')\
                .where('owner_uid', '==', user_uid)\
                .where('is_hosted', '==', True)\
                .stream()
            
            for doc in my_instances:
                data = doc.to_dict()
                parent_id = data.get('parent_bot_id')
                if parent_id:
                    active_sessions[parent_id] = doc.id
                    
        except Exception as e:
            print(f"Auth check error in public_bots: {e}")

    response = make_response(render_template(
        "public_bots.html", 
        bots=bots, 
        active_sessions=active_sessions
    ))
    
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route("/my-bots")
@login_required 
def my_bots():
    bots = []
    user_uid = g.user_uid 
    try:
        # Fetch bots owned by user
        bots_ref = db.collection('rooms').where('owner_uid', '==', user_uid).limit(50)
        
        for doc in bots_ref.stream():
            bot_data = doc.to_dict()
            
            # === FILTER LOGIC ===
            # 1. Check explicit flag (for new bots we create from now on)
            if bot_data.get('is_hosted') == True:
                continue
                
            # 2. Check heuristic (for existing bots)
            # Hosted instances usually have a specific starting system message
            messages = bot_data.get('messages', [])
            if messages and len(messages) > 0:
                first_msg_text = messages[0].get('text', '')
                # If the first message says "Instance created", it's a hosted game, not a bot definition
                if "Instance created" in first_msg_text:
                    continue
            # ====================

            bot_data['id'] = doc.id
            bots.append(bot_data)
            
    except Exception as e:
        print(f"Error: {e}")
        
    return render_template("my_bots.html", bots=bots)

@app.route("/generate-bot-image", methods=["POST"])
@login_required
def generate_bot_image():
    if not stability_api_key: return jsonify({'success': False, 'error': 'No API Key'}), 500
    try:
        data = request.get_json()
        prompt = f"A beautiful portrait of a {data.get('age')} year old {data.get('gender')}, {data.get('appearance')}. digital art, anime style"
        answers = stability_api.generate(
            prompt=[generation.Prompt(text=prompt, parameters=generation.PromptParameters(weight=1.0))],
            style_preset="anime", steps=30, cfg_scale=7.0, width=512, height=512, samples=1
        )
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    upload_result = cloudinary.uploader.upload(artifact.binary, folder="bot_avatars", public_id=f"bot_{generate_room_code(10)}")
                    return jsonify({'success': True, 'image_url': upload_result.get('secure_url')})
        return jsonify({'success': False, 'error': 'No image generated'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/upload-bot-image", methods=["POST"])
@login_required
def upload_bot_image():
    if 'file' not in request.files: return jsonify({'success': False, 'error': 'No file'}), 400
    file = request.files['file']
    if file:
        try:
            res = cloudinary.uploader.upload(file, folder="bot_avatars", public_id=f"bot_{generate_room_code(10)}")
            return jsonify({'success': True, 'image_url': res.get('secure_url')})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False}), 400

@app.route("/create", methods=["GET", "POST"])
@login_required 
def create_room():
    if request.method == "POST":
        room_code = generate_room_code()
        user_uid = g.user_uid 
        user_name = g.user.get('name', g.user.get('email', 'Anonymous'))
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc) 
        
        new_room = {
            'bot_name': request.form.get('bot_name', 'Bot'),
            'bot_personality': request.form.get('bot_personality', 'friendly'),
            'start_scenario': request.form.get('start_scenario', 'Hello'),
            'difficulty': int(request.form.get('difficulty', 5)),
            'game_over': False,
            'users': {}, 
            'messages': [
                {'user_id': 'System', 'text': f"Bot created by {user_name}.", 'timestamp': py_time},
                {'user_id': request.form.get('bot_name'), 'text': request.form.get('start_scenario'), 'timestamp': py_time}
            ],
            'bot_image_url': request.form.get('bot_image_url'),
            'bot_appearance': request.form.get('appearance'),
            'owner_uid': user_uid,
            'owner_display_name': user_name,
            'is_public': request.form.get('is_public') == 'on',
            'model_version': 'flash',
            'is_hosted': False  # <--- NEW FLAG: It's an original bot
        }
        db.collection('rooms').document(room_code).set(new_room)
        return redirect(url_for('my_bots'))
    return render_template("create.html")

@app.route("/join", methods=["GET", "POST"])
@login_required 
def join_room():
    # Support both GET (from link) and POST (from our new Modal)
    room_code = request.args.get('code') or request.form.get('room_code')
    mode = request.args.get('mode') or request.form.get('mode')
    
    # New Form Data
    privacy = request.form.get('privacy', 'public')
    initial_model = request.form.get('initial_model', 'flash')

    if not room_code: return redirect(url_for('index'))

    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists:
        flash("Room not found.")
        return redirect(url_for('index'))
    
    room_data = room_doc.to_dict()
    user_uid = g.user_uid
    user_name = g.user.get('name', g.user.get('email', 'Player'))
    
    # === HOST MODE (Creating a new Instance) ===
    if mode == 'new':
        new_code = generate_room_code()
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        
        # Determine settings based on privacy choice
        is_public_game = (privacy == 'public')
        
        if is_public_game:
            # Public Game: Defaults to Flash, switches on Win, No fixed model
            chosen_model = 'flash'
            fixed_model = False
        else:
            # Private Game: User chooses model, Flag set to True
            is_public_game = False
            chosen_model = initial_model # 'flash' or 'pro'
            fixed_model = True

        cloned_data = {
            'bot_name': room_data.get('bot_name'),
            'bot_personality': room_data.get('bot_personality'),
            'start_scenario': room_data.get('start_scenario'),
            'difficulty': room_data.get('difficulty', 5),
            'game_over': False,
            'winner_uid': None,
            'users': { user_uid: {'nickname': user_name, 'score': 0} }, 
            'messages': [
                {'user_id': 'System', 'text': f"Instance created. Mode: {privacy.capitalize()}.", 'timestamp': py_time},
                {'user_id': room_data.get('bot_name'), 'text': room_data.get('start_scenario'), 'timestamp': py_time}
            ],
            'bot_image_url': room_data.get('bot_image_url'),
            'bot_appearance': room_data.get('bot_appearance'),
            'owner_uid': user_uid,
            'owner_display_name': user_name,
            'is_public': is_public_game,       # Public vs Private
            'model_version': chosen_model,     # Selected Model
            'fixed_model': fixed_model,        # NEW FLAG
            'allowed_viewers': [],
            'parent_bot_id': room_code,
            'is_hosted': True 
        }
        db.collection('rooms').document(new_code).set(cloned_data)
        return redirect(url_for('chat_room', room_code=new_code))

    # === NORMAL JOIN (Existing logic unchanged) ===
    if user_uid not in room_data.get('users', {}):
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        room_ref.update({
            f'users.{user_uid}': {'nickname': user_name, 'score': 0}, 
            'messages': firestore.ArrayUnion([{
                'user_id': 'System', 'text': f"{user_name} joined.", 'timestamp': py_time
            }])
        })
        
    return redirect(url_for('chat_room', room_code=room_code))

# === CHAT: Access Control & Game Logic ===
@app.route("/room/<room_code>", methods=["GET", "POST"])
@login_required
def chat_room(room_code):
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists: return redirect(url_for('index'))
    
    room_data = room_doc.to_dict()
    user_uid = g.user_uid
    
    if user_uid not in room_data.get('users', {}):
        return redirect(url_for('join_room', code=room_code))
        
    user_data = room_data['users'][user_uid]
    nickname = user_data.get('nickname', 'Player')
    is_owner = (room_data.get('owner_uid') == user_uid)
    is_winner = (room_data.get('winner_uid') == user_uid)

    # Privacy Check
    has_access = True
    winner_uid = room_data.get('winner_uid')
    if winner_uid:
        allowed = room_data.get('allowed_viewers', [])
        # Winner, Owner, or Allowed users can see/chat
        if user_uid != winner_uid and user_uid not in allowed:
            has_access = False

    if request.method == "POST":
        if not has_access: return jsonify({'success': False, 'error': 'Locked'}), 403
        
        msg = request.form.get('message')
        if not msg: return jsonify({'success': False}), 400
            
        try:
            py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
            msgs_to_add = [{'user_id': user_uid, 'text': msg, 'timestamp': py_time}]
            
            # PRO MODE (Unfiltered/Winner)
            if room_data.get('model_version') == 'pro':
                if not openrouter_api_key: raise ValueError("No OpenRouter Key")
                
                hist = build_openrouter_prompt_messages(room_data, nickname, msg)
                
                # Make the request
                r = requests.post(
                    OPENROUTER_API_URL, 
                    headers={
                        "Authorization": f"Bearer {openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://botafection.com", # Required by OpenRouter
                        "X-Title": "BotAffection" # Required by OpenRouter
                    },
                    json={"model": PRO_MODEL_NAME, "messages": hist}
                )
                
                response_json = r.json()
                
                # --- ERROR CHECKING ---
                if 'error' in response_json:
                    print(f"OpenRouter API Error: {response_json}")
                    error_msg = response_json['error'].get('message', 'Unknown API Error')
                    raise ValueError(f"AI Provider Error: {error_msg}")
                
                if 'choices' not in response_json:
                    print(f"Unexpected Response: {response_json}")
                    raise ValueError("AI response format was invalid.")
                # ----------------------

                bot_resp = response_json['choices'][0]['message']['content']
                msgs_to_add.append({'user_id': room_data['bot_name'], 'text': bot_resp, 'timestamp': py_time})
                room_ref.update({'messages': firestore.ArrayUnion(msgs_to_add)})
            
            # GAME MODE
            else:
                prompt = build_gemini_prompt(room_data, nickname, msg)
                res = model_flash.generate_content(prompt)
                parsed = parse_gemini_response(res.text)
                
                new_score = max(0, min(100, user_data['score'] + parsed['affection_change']))
                update_ops = { f'users.{user_uid}.score': new_score }
                
                msgs_to_add.append({'user_id': room_data['bot_name'], 'text': parsed['response'], 'timestamp': py_time})
                
                # Win Condition
                if new_score >= 100 and not winner_uid:
                    update_ops['winner_uid'] = user_uid
                    update_ops['winner_name'] = nickname
                    update_ops['allowed_viewers'] = [user_uid]
                    
                    # === MODIFIED LOGIC ===
                    # Only switch to Pro if the model is NOT fixed
                    if not room_data.get('fixed_model', False):
                        update_ops['model_version'] = 'pro'
                        msgs_to_add.append({'user_id': 'System', 'text': f"🏆 {nickname} won! Unlocked Pro Model.", 'timestamp': py_time})
                    else:
                        # Private/Fixed game - acknowledge win but no model switch
                        msgs_to_add.append({'user_id': 'System', 'text': f"🏆 {nickname} won! You have conquered this timeline.", 'timestamp': py_time})
                    # ======================

                update_ops['messages'] = firestore.ArrayUnion(msgs_to_add)
                room_ref.update(update_ops)

            return jsonify({'success': True})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template("room.html", 
                           room=room_data, 
                           room_code=room_code, 
                           user_id=user_uid, 
                           is_owner=is_owner,
                           is_winner=is_winner,
                           has_access=has_access)

# === NEW: HOST ACTIONS ===
@app.route("/reset-chat", methods=["POST"])
@login_required
def reset_chat():
    code = request.form.get('room_code')
    room_ref = db.collection('rooms').document(code)
    doc = room_ref.get()
    
    if not doc.exists: return jsonify({'success': False}), 404
    
    room_data = doc.to_dict()
    
    # Only Owner (Host) can reset
    if room_data.get('owner_uid') != g.user_uid:
        return jsonify({'success': False, 'error': 'No permission'}), 403
        
    # === NEW LOGIC: HOSTED vs ORIGINAL ===
    
    # Case 1: It is a Game Session (Private/Public instance you created)
    if room_data.get('is_hosted', False):
        # Delete the room entirely to "break" the Resume link
        room_ref.delete()
        flash("Session ended. You can start a new game with new settings.")
        return redirect(url_for('public_bots'))
        
    # Case 2: It is an Original Bot (You are the creator of the bot itself)
    else:
        # Just clear the history, do NOT delete the bot
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        bot_name = room_data.get('bot_name')
        start_scen = room_data.get('start_scenario')
        
        updates = {
            'messages': [
                {'user_id': 'System', 'text': "Chat reset by Owner.", 'timestamp': py_time},
                {'user_id': bot_name, 'text': start_scen, 'timestamp': py_time}
            ],
            'winner_uid': None,
            'winner_name': None,
            'model_version': 'flash',
            'allowed_viewers': []
        }
        # Reset scores
        for uid in room_data.get('users', {}):
            updates[f'users.{uid}.score'] = 0
            
        room_ref.update(updates)
        return redirect(url_for('chat_room', room_code=code))

# === NEW: ACCESS REQUESTS ===
@app.route("/request-access", methods=["POST"])
@login_required
def request_access():
    code = request.form.get('room_code')
    name = g.user.get('name', 'User')
    db.collection('rooms').document(code).update({
        'access_requests': firestore.ArrayUnion([{'uid': g.user_uid, 'name': name}])
    })
    return jsonify({'success': True})

@app.route("/grant-access", methods=["POST"])
@login_required
def grant_access():
    code = request.form.get('room_code')
    target = request.form.get('target_uid')
    
    # Verify requester is Winner
    room_ref = db.collection('rooms').document(code)
    if room_ref.get().to_dict().get('winner_uid') != g.user_uid:
        return jsonify({'success': False}), 403
        
    room_ref.update({'allowed_viewers': firestore.ArrayUnion([target])})
    return jsonify({'success': True})

@app.route("/delete-bot", methods=["POST"])
@login_required
def delete_bot():
    room_code = request.form.get('room_code')
    if not room_code:
        flash("Error: No room code provided.")
        return redirect(url_for('my_bots'))
    
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()
    
    if room_doc.exists:
        # Check ownership
        if room_doc.to_dict().get('owner_uid') == g.user_uid:
            room_ref.delete()
            flash("Bot deleted successfully.")
        else:
            flash("Error: You do not have permission to delete this bot.")
    else:
        flash("Error: Bot not found.")
        
    return redirect(url_for('my_bots'))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
