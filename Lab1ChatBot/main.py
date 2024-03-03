"""
  ------------- LAB 1 - INTELIGENCIA ARTIFICIAL E ROBOTICA -----------------
           Matheus Venâncio Scomparim      - R.A: 22.121.063-6
           Gustavo Bueno Leite de Olvieira - R.A: 22.121.057-8
  --------------------------------------------------------------------------
"""


from chatbot import ChatBot
myChatBot = ChatBot()
#apenas carregar um modelo pronto
#myChatBot.loadModel()

#criar o modelo
myChatBot.createModel()

print("\nBem vindo ao Chatbot - PIPE")


pergunta = input("Como posso te ajudar?\n")
resposta, intencao = myChatBot.chatbot_response(pergunta)
print(resposta + "   ["+intencao[0]['intent']+"]")


while (intencao[0]['intent']!="despedida"):
    pergunta = input("\nPosso lhe ajudar com mais alguma coisa?\n")
    resposta, intencao = myChatBot.chatbot_response(pergunta)
    print(resposta + "   [" + intencao[0]['intent'] + "]")

print("\nFoi um prazer atendê-lo")
