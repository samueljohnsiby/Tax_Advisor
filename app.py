import streamlit as st
from bot import TaxAdvisor
from document_loader import GetContext

from dotenv import dotenv_values

env_vars = dotenv_values(".env")
class ChatApplication:
    def __init__(self) -> None:
        self.system_intel = """You are a tax advisor in India. You will be asked questions about the tax system from individuals,
                            not corporations. Please assist the users with their queries using your knowledge of the tax laws in India.
                            You will be provided context from the Income Tax Law of India. Please use this context to craft your answers.Create inferences from the context given. 
                            Whatever comes after context, you can use to generate your answer even in the history. If context is not provided for a 
                            particular user query, 
                            please check if the question is related to taxation in India, if so check history for the required context., 
                            Incase the question is not about taxation please ask user to ask questions about income tax rules of india.
                            If you dont have enough context to answer the question even from the history say you are unable to help. 
                            Never provide answers where context is not available to support your answer
                            Never refer to the context provided as end user is unable to see it
                          """

        self.api_key = env_vars.get("API_KEY")
        self.tax_advisor = TaxAdvisor(self.api_key)
        self.getcontext = GetContext()

    def run(self):
        
        st.title("Tax Advisor")
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):

                    st.markdown(message["content"].split("context:")[0].strip())

        st.session_state.messages.append({"role": "system", "content": self.system_intel})

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            context = self.getcontext.get_context(prompt)

            prompt_with_context = (
                f"{prompt} context: Please use the below context to provide answer to the user query keeping in mind system prompt {context}"
            )
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt_with_context}
            )

            # Get response from the bot
            response = self.tax_advisor.chat_with_user(st.session_state.messages)


            # Display bot response in chat message container
            with st.chat_message("bot"):
                st.markdown(response)

            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':

    if 'chat_app' not in st.session_state:
        st.session_state.chat_app = ChatApplication()

    # chat_application = ChatApplication()
    st.session_state.chat_app.run()
