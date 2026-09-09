import uuid

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1/chat"
REQUEST_TIMEOUT = 120

st.set_page_config(
    page_title="Real Estate AI Agent",
    page_icon="🏠",
)

st.title("🏠 Real Estate AI Agent")
st.caption("Interrogez les données immobilières DVF de Paris.")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Affichage de l'historique
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

message = st.chat_input(
    "Exemple : Quel est le prix médian au m² dans le 15e en 2024 ?"
)

if message:
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyse des données immobilières..."):
                response = requests.post(
                    API_URL,
                    json={
                        "message": message,
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                response_data = response.json()
                answer = response_data["answer"]

            st.markdown(answer)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de contacter l’API. "
                "Vérifie que FastAPI est lancé sur le port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "L’agent a dépassé le délai maximal de réponse."
            )

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code
            error_message = exc.response.text

            st.error(
                f"Erreur API {status_code} : {error_message}"
            )

        except requests.exceptions.JSONDecodeError:
            st.error(
                "La réponse reçue n’est pas un JSON valide."
            )

        except KeyError:
            st.error(
                "La réponse de l’API ne contient pas le champ 'answer'."
            )