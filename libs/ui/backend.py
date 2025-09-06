from flask import Flask, request, jsonify

def create_backend_server(model, role_name, roles_sentences, init_role_by_name):
    app = Flask(__name__)
    role_state = {"current_role": role_name, "available_roles": list(roles_sentences.keys())}

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        user_input = data.get("message", "")
        try : 
            reply = model.chat(user_input)
            return jsonify({"reply": reply})
        except Exception as e:
            print(e)

    @app.route("/role_state", methods=["GET"])
    def get_role_state():
        try:
            current_role_name = getattr(model, "role_name", None)
            return jsonify({
                "current_role": role_state.get("current_role", None),
                "available_roles": list(roles_sentences.keys()),
            })
        except Exception as e:
            print("Failed to get role state.")
            return jsonify({"error": str(e)}), 500

    @app.route("/role_info/<role_name>", methods=["GET"])
    def get_role_info(role_name):
        try:
            if role_name not in roles_sentences:
                return jsonify({"error": f"Role '{role_name}' not found."}), 404
            
            sentences_count = len(roles_sentences[role_name])
            return jsonify({
                "role_name": role_name,
                "sentences_count": sentences_count,
                "sample_sentences": roles_sentences[role_name][:3] if sentences_count > 0 else []
            })
        except Exception as e:
            return jsonify({"error": f"Failed to get role info: {str(e)}"}), 500

    @app.route("/switch_role", methods=["POST"])
    def switch_role():
        data = request.get_json()
        role_names = data.get("roles")
        if not role_names:
            return jsonify({"error": "No roles provided."}), 400
        if isinstance(role_names, str):
            role_names = [role_names]
        try:
            new_role_name = init_role_by_name(role_names)
            role_state["current_role"] = new_role_name
            return jsonify({"message": f"Switched to role: {new_role_name}", "current_role": new_role_name})
        except Exception as e:
            print(f"Failed to switch role: {str(e)}")
            return jsonify({"error": f"Failed to switch role: {str(e)}"}), 500

    return app
