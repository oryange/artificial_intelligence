import streamlit as st
import pandas as pd
from groq import Groq
import json
import matplotlib.pyplot as plt
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_preco_acao(ticker):
    print(ticker)
    return str(yf.Ticker(ticker).history(period="1y").iloc[-1]["Close"])


def plot_preco_acao(ticker):
    data = yf.Ticker(ticker).history(period="1y")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="Preço de Fechamento")
    plt.title(f"Preço da Ação {ticker} nos Últimos 12 Meses")
    plt.xlabel("Data")
    plt.ylabel("Preço (R$)")
    plt.legend()
    plt.grid(True)
    os.makedirs("./images", exist_ok=True)
    plt.savefig(f"./images/{ticker}.png")
    plt.close()


funcoes = [
    {
        "type": "function",
        "function": {
            "name": "get_preco_acao",
            "description": "Retorna o preço atual de uma ação com base no ticker fornecido.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "O ticker da ação (ex: AAPL, TSLA, AMZN).",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_preco_acao",
            "description": "Gera um gráfico do preço da ação ao longo do último ano com base no ticker fornecido.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "O ticker da ação (ex: AAPL, TSLA, AMZN).",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
]

available_functions = {
    "get_preco_acao": get_preco_acao,
    "plot_preco_acao": plot_preco_acao,
}

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": "Você é um assistente de investimento que auxilia no fornecimento de informações sobre ações brasileiras.",
        }
    ]

st.title("Assistente Financeiro de Ações")

user_input = st.text_input("Digite sua pergunta sobre ações brasileiras:")

if user_input:
    try:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state["messages"],
            tools=funcoes,
            tool_choice="auto",
        )
        response_message = response.choices[0].message

        if response_message.tool_calls:
            print(response_message)
            print(type(response_message))

            function_name = response_message.tool_calls[0].function.name
            function_args = json.loads(response_message.tool_calls[0].function.arguments)
            tool_call_id = response_message.tool_calls[0].id

            print(function_name, function_args)

            function_response = available_functions[function_name](**function_args)

            if function_name == "plot_preco_acao":
                st.image(f"./images/{function_args['ticker']}.png")
            else:
                st.session_state["messages"].append(response_message)
                st.session_state["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": function_response,
                    }
                )

                second_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=st.session_state["messages"],
                )
                print(second_response.choices[0].message.content)
                st.text(second_response.choices[0].message.content.strip())
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": second_response.choices[0].message.content.strip(),
                    }
                )
        else:
            st.text(response_message.content)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": response_message.content,
                }
            )
    except Exception as e:
        st.text("tente novamente mais tarde")
        print(e)
