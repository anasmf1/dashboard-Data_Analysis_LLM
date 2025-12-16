import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Charge les variables d'environnement (API KEY)
load_dotenv()

# Initialisation du client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(
    user_prompt: str, 
    system_context: str, 
    conversation_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.3, 
    model: str = "gpt-4o-mini" # Ou "gpt-3.5-turbo" selon ton budget
) -> str:
    """
    Fonction générique pour appeler l'IA.
    Gère l'historique et le contexte système.
    """
    try:
        messages = [{"role": "system", "content": system_context}]
        
        # Ajout de l'historique (limité aux 6 derniers échanges)
        if conversation_history:
            # On s'assure que l'historique a le bon format pour l'API
            clean_history = [
                {"role": m["role"], "content": m["content"]} 
                for m in conversation_history[-6:]
            ]
            messages.extend(clean_history)
        
        # Ajout du message actuel de l'utilisateur
        messages.append({"role": "user", "content": user_prompt})
        
        # Appel API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Erreur OpenAI: {e}")
        return "⚠️ Désolé, je n'arrive pas à générer la formule pour le moment. Vérifiez votre clé API."