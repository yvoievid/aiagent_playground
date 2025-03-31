from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from agent.state import State

load_dotenv()


class LLMOptimizer:
    def __init__(self):
        self.url = "https://llvm.org/docs/LangRef.html"
        self.chat_inst = init_chat_model("o3-mini", model_provider="openai")
        self.docs = None
        self.vector_store = None

        self.generation_prompt_template = PromptTemplate.from_template("""
            You are an AI that optimizes LLVM IR code using provided documentation.
            Use the following Python libraries:
              - llvmlite==0.44.0
              - numba==0.61.0
              - numpy==2.1.3
            Ensure the output has:
              - No LLVM IR parsing errors
              - No comments or additional text
              - Output only valid LLVM IR starting with 'define i32'
              - Compile it before responding to ensure there is no errors
             - Preserve the comments

            DON'T rename function names or variables  
            DON'T improve readability, ONLY optimize calculations  
            DON'T remove the comments  
            DON'T Change patched function name
            DON'T Change function signature 
            {context}

            Use this code as example of optimization:

            {example}
            llvmir: {question}
            Keep the following kind of function names unchanged:
            _ZN8problems8problem36matmulB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEiii5ArrayIfLi2E1A7mutable7alignedE5ArrayIfLi2E1A7mutable7alignedE
        """)

        self.validation_prompt_template = PromptTemplate.from_template("""
            You generated this LLVM IR using this documentation:

            {context}   

            Code to refactor:
            {llvmir}

            Cache common calculations to reduce redundant operations  
            Restructure loops and condition logic for more efficient execution  

            Please regenerate the IR, making sure:
            - All values in phi instructions are properly defined in the corresponding predecessor blocks  
            - No LLVM IR parsing errors or undefined variables  
            - No runtime errors

            Example error to catch:
                RuntimeError: LLVM IR parsing error  
                <string>:133:52: error: use of undefined value '%B44.us.us.us'

                RuntimeError: LLVM IR parsing error  
                <string>:70:37: error: expected '(' in logical constantexpr  
                skip_compute = or i1 %m_zero, or i1 %n_too_small, %k_zero

                ValueError: Optimized IR should end with a newline followed by a closing brace 
                
            Only return the fixed LLVM IR.  
            DON'T rename function names or variables  
            DON'T improve readability, ONLY optimize calculations  
            DON'T Change function name
            DON'T Change function signature
            - Preserve all input and output shapes
            - Preserve the comments

            Use this code as example of optimization:

            {example}
            Start the response immediately with '; Function Attrs ' and do not add any quotes.
            Keep the following kind of function names unchanged:
            _ZN8problems8problem36matmulB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEiii5ArrayIfLi2E1A7mutable7alignedE5ArrayIfLi2E1A7mutable7alignedE
       
            """)

    def optimize(self, llvmir: str) -> str:
        print("Start LLM optimization")
        self._load_llvm_docs()
        
        with open("agent/example_optimized_llvm.txt", "r") as f:
            optimized_code = f.read()

        graph_builder = StateGraph(State).add_sequence([
            self._retrieve,
            self._generate,
            self._validate
        ])
        graph_builder.add_edge(START, "_retrieve")
        graph = graph_builder.compile()

        response = graph.invoke({"question": llvmir, "example": optimized_code})

        print("End LLM optimization")
        return response["answer"]

    def _load_llvm_docs(self):
        loader = WebBaseLoader(web_paths=[self.url])
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(embeddings)
        _ = self.vector_store.add_documents(documents=all_splits)

    def _retrieve(self, state: State) -> State:
        retrieved_docs = self.vector_store.similarity_search(state["question"], k=10)
        return {"context": retrieved_docs}

    def _generate(self, state: State) -> State: 
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = self.generation_prompt_template.invoke({
            "question": state["question"],
            "context": docs_content,
            "example": state["example"]
        })

        response = self.chat_inst.invoke(prompt)
        return {"answer": response.content}

    def _validate(self, state: State) -> State:
        prompt = self.validation_prompt_template.invoke({
            "llvmir": state["answer"],
            "context": state["context"],
            "example": state["example"]
        })

        response = self.chat_inst.invoke(prompt)
        return {"answer": response.content}
