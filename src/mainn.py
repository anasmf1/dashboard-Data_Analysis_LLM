import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# Import du module qu'on vient de créer
from LLM1 import call_llm

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Assistant DAX Sales", version="1.0.0")

# Autoriser le frontend HTML à communiquer avec ce backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# --- 1. DÉFINITION DU MODÈLE DE DONNÉES (TON CAS SPÉCIFIQUE) ---
# ==============================================================================

SALES_SCHEMA = """
Tu es un expert Power BI & DAX. Tu dois générer des mesures DAX précises.

--- MODÈLE DE DONNÉES ---

Table 1 : 'Details' (Table de faits - Transactions)
- Colonnes : [Amount], [Category], [Order ID] (Clé étrangère), [PaymentMode], [Profit], [Quantity], [Sub-Category]

Table 2 : 'Orders' (Table de dimension - Infos Commande)
- Colonnes : [Order ID] (Clé primaire), [City], [CustomerName], [Order Date], [State]

RELATION :
'Orders'[Order ID] (1) <---> (*) 'Details'[Order ID]
(La table Orders filtre la table Details).

--- RÈGLES DAX ---
1. Utilise toujours la syntaxe anglaise standard (SUM, CALCULATE, DIVIDE, RELATED).
2. Pour les ratios, utilise toujours DIVIDE(numérateur, dénominateur, 0).
3. Si on demande une variation temporelle (YTD, YoY), utilise la colonne 'Orders'[Order Date].
4. Formate le code avec des variables (VAR / RETURN) pour la lisibilité.
5. N'invente pas de colonnes qui ne sont pas dans la liste ci-dessus.
"""

# ==============================================================================
# --- 2. GESTION DES REQUÊTES ---
# ==============================================================================

class Message(BaseModel):
    role: str # "user" ou "assistant"
    content: str

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[Message]] = []
    # On garde system_instruction optionnel si le frontend veut surcharger, 
    # sinon on utilise notre SALES_SCHEMA par défaut
    system_instruction: Optional[str] = None 

@app.get("/health")
def health_check():
    return {"status": "online", "system": "Sales DAX Engine Ready"}

@app.post("/chat")
async def generate_response(request: ChatRequest):
    try:
        logger.info(f"📩 Question reçue : {request.prompt}")
        
        # Conversion de l'historique Pydantic en liste de dicts
        history_dicts = [m.model_dump() for m in request.history]

        # Détermination du contexte système
        # Si le frontend envoie une instruction spécifique, on l'utilise, sinon on prend le schéma par défaut
        context = request.system_instruction if request.system_instruction else SALES_SCHEMA

        # Appel au module IA
        reply = call_llm(
            user_prompt=request.prompt,
            system_context=context,
            conversation_history=history_dicts,
            temperature=0.2 # Température basse pour avoir du code rigoureux
        )
        
        return {"reply": reply}

    except Exception as e:
        logger.error(f"Erreur serveur : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    print("🚀 Serveur DAX Sales démarré sur http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)