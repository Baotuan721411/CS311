import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from src.utils import TrainQuestion, TestQuestion, Topic, Document
from rank_bm25 import BM25Okapi
import json
from typing import List, Dict, Tuple
import torch
import math
import re 
def simple_tokenize(text):
    return re.sub(r"[^\w\s]", " ", str(text).lower()).split() if text else []

class RetrieverBase:
    def __init__(
        self,
        train_docs_path: str,
        train_questions_path: str,
        dev_docs_path: str,
        dev_questions_path: str,
        test_docs_path: str,
        test_questions_path: str,
        prompt_path: str,
        train_output_path: str,
        dev_output_path: str,
        test_output_path: str,
        query_choice: str,
        content_choice: str,
        top_k: int
    ):
        self.train_docs_path = train_docs_path
        self.train_questions_path = train_questions_path
        self.dev_docs_path = dev_docs_path
        self.dev_questions_path = dev_questions_path
        self.test_docs_path = test_docs_path
        self.test_questions_path = test_questions_path
        self.top_k = top_k
        self.train_output_path = train_output_path
        self.dev_output_path = dev_output_path
        self.test_output_path = test_output_path
        self.query_choice = query_choice
        self.content_choice = content_choice
        self.prompt_template = self.load_prompt_template(prompt_path)
        self.extract_data()
        
    def _load_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_jsonl(self, path: str) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def extract_data(self) -> Tuple[List[Topic], List[TrainQuestion]]:
        Train_topics: List[Topic] = []
        Train_questions: List[TrainQuestion] = []
        Dev_topics: List[Topic] = []
        Dev_questions: List[TrainQuestion] = []
        Test_topics: List[Topic] = []
        Test_questions: List[TestQuestion] = []
        
        train_docs = self._load_json(self.train_docs_path)
        dev_docs = self._load_json(self.dev_docs_path)
        test_docs = self._load_json(self.test_docs_path)
        
        train_questions = self._load_jsonl(self.train_questions_path)
        dev_questions = self._load_jsonl(self.dev_questions_path)
        test_questions = self._load_jsonl(self.test_questions_path)
        
        for topic in train_docs:
            topic_id = topic.get("topic_id")
            docs = []
            for doc in topic.get("docs", []):
                docs.append(
                    Document(
                        doc_id=doc.get("id"),
                        title=doc.get("title", ""),
                        snippet=doc.get("snippet", ""),
                        content=doc.get("content", "")
                    )
                )
            Train_topics.append(
                Topic(
                    topic_id = topic_id,
                    docs = docs
                )
            )
        for q in train_questions:
            Train_questions.append(
                TrainQuestion(
                    question_id = q.get("id"),
                    topic_id = q.get("topic_id"),
                    target_event = q.get("target_event", ""),
                    option_A = q.get("option_A"),
                    option_B = q.get("option_B"),
                    option_C = q.get("option_C"),
                    option_D = q.get("option_D"),
                    golden_answer = q.get("golden_answer")
                )
            )
        self.train_data = (Train_topics, Train_questions)
        
        for topic in dev_docs:
            topic_id = topic.get("topic_id")
            docs = []
            for doc in topic.get("docs", []):
                docs.append(
                    Document(
                        doc_id=doc.get("id"),
                        title=doc.get("title", ""),
                        snippet=doc.get("snippet", ""),
                        content=doc.get("content", "")
                    )
                )
            Dev_topics.append(
                Topic(
                    topic_id = topic_id,
                    docs = docs
                )
            )
        for q in dev_questions:
            Dev_questions.append(
                TrainQuestion(
                    question_id = q.get("id"),
                    topic_id = q.get("topic_id"),
                    target_event = q.get("target_event", ""),
                    option_A = q.get("option_A"),
                    option_B = q.get("option_B"),
                    option_C = q.get("option_C"),
                    option_D = q.get("option_D"),
                    golden_answer = q.get("golden_answer")
                )
            )
        self.dev_data = (Dev_topics, Dev_questions)
        
        for topic in test_docs:
            topic_id = topic.get("topic_id")
            docs = []
            for doc in topic.get("docs", []):
                docs.append(
                    Document(
                        doc_id=doc.get("id"),
                        title=doc.get("title", ""),
                        snippet=doc.get("snippet", ""),
                        content=doc.get("content", "")
                    )
                )
            Test_topics.append(
                Topic(
                    topic_id = topic_id,
                    docs = docs
                )
            )
        for q in test_questions:
            Test_questions.append(
                TestQuestion(
                    question_id = q.get("id"),
                    topic_id = q.get("topic_id"),
                    target_event = q.get("target_event", ""),
                    option_A = q.get("option_A"),
                    option_B = q.get("option_B"),
                    option_C = q.get("option_C"),
                    option_D = q.get("option_D"),
                )
            )
        self.test_data = (Test_topics, Test_questions)

    def retrieve(self):
        pass
    
    @staticmethod
    def load_prompt_template(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def render_prompt(template, variables):
        for k, v in variables.items():
            template = template.replace(f"{{{{{k}}}}}", str(v))
        return template  
    
    @staticmethod
    def build_retrieval_query_from_options (question_entry):
        parts = []
        parts.append("possible causes of an event:")
        for opt_key in ["option_A", "option_B", "option_C", "option_D"]:
            opt = question_entry.get(opt_key)
            if opt:
                parts.append(str(opt))
        return " ".join(parts)
    
    @staticmethod
    def normalize_scores(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if math.isclose(mx, mn):
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)  
            
                
class BM25Retriever(RetrieverBase):
    def __init__(
        self,
        model_name,
        alpha,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.model_name = model_name #We do this just to fix error. No real use
        
    def retrieve(self):
        topic_indices = {}
        for topic in self.train_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            docs_pass_to_BM25 = []
            stored_docs = []
            
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_BM25.append(simple_tokenize(retrieval_text)) 
                         
            bm25 = BM25Okapi(docs_pass_to_BM25)
            
            topic_indices[topic_id] = {
                "bm25": bm25,
                "docs": stored_docs,
            }
        
        formatted_data = []  
        for question in self.train_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            query_tokens = simple_tokenize (query_text)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]
            bm25_scores = topic_index["bm25"].get_scores(query_tokens)
            bm25_norm = self.normalize_scores(bm25_scores)

            top_indices = np.argsort(bm25_norm)[:self.top_k]
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            raw_ans = question.golden_answer
            assistant_response = ", ".join(raw_ans) if isinstance(raw_ans, list) else str(raw_ans)
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            })
        with open(self.train_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.train_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        topic_indices = {}
        for topic in self.dev_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            docs_pass_to_BM25 = []
            stored_docs = []
            
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_BM25.append(simple_tokenize(retrieval_text)) 
                        
            bm25 = BM25Okapi(docs_pass_to_BM25)
            
            topic_indices[topic_id] = {
                "bm25": bm25,
                "docs": stored_docs,
            }
        
        formatted_data = []  
        for question in self.dev_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            query_tokens = simple_tokenize (query_text)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]
            bm25_scores = topic_index["bm25"].get_scores(query_tokens)
            bm25_norm = self.normalize_scores(bm25_scores)

            top_indices = np.argsort(bm25_norm)[:self.top_k]
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            raw_ans = question.golden_answer
            assistant_response = ", ".join(raw_ans) if isinstance(raw_ans, list) else str(raw_ans)
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            })
        with open(self.dev_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.dev_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        topic_indices = {}
        for topic in self.test_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            docs_pass_to_BM25 = []
            stored_docs = []
            
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_BM25.append(simple_tokenize(text)) 
                         
            bm25 = BM25Okapi(docs_pass_to_BM25)
            
            topic_indices[topic_id] = {
                "bm25": bm25,
                "docs": stored_docs,
            }
        
        formatted_data = []  
        for question in self.test_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            query_tokens = simple_tokenize (query_text)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]
            bm25_scores = topic_index["bm25"].get_scores(query_tokens)
            bm25_norm = self.normalize_scores(bm25_scores)

            top_indices = np.argsort(bm25_norm)[:self.top_k]
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )

            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            })
        with open(self.test_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.test_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        return len(formatted_data)
    


class SBertRetriever(RetrieverBase):
    def __init__(
        self,
        model_name: str,
        alpha: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.model = SentenceTransformer(model_name)
    def retrieve(self):
        topic_indices = {}
        for topic in self.train_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            stored_docs = []
            docs_pass_to_Sbert = []
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_Sbert.append(retrieval_text)   
                         
            embeddings = self.model.encode(docs_pass_to_Sbert, convert_to_tensor=True, show_progress_bar=False)
            

            topic_indices[topic_id] = {
                "docs": stored_docs,
                "embeddings": embeddings
            }
        
        formatted_data = []  
        for question in self.train_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]

            query_embedding = self.model.encode (query_text, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = cos_sim (query_embedding, topic_index ["embeddings"])[0]

            _, top_indices = torch.topk(cosine_scores, k = self.top_k, largest=True)
            top_indices = top_indices.tolist()
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            raw_ans = question.golden_answer
            assistant_response = ", ".join(raw_ans) if isinstance(raw_ans, list) else str(raw_ans)
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            })
        with open(self.train_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.train_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        topic_indices = {}
        for topic in self.dev_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            stored_docs = []
            docs_pass_to_Sbert = []
            
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_Sbert.append(retrieval_text)   
                         
            embeddings = self.model.encode(docs_pass_to_Sbert, convert_to_tensor=True, show_progress_bar=False)
            

            topic_indices[topic_id] = {
                "docs": stored_docs,
                "embeddings": embeddings
            }
        
        formatted_data = []  
        for question in self.dev_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]

            query_embedding = self.model.encode (query_text, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = cos_sim (query_embedding, topic_index ["embeddings"])[0]

            _, top_indices = torch.topk(cosine_scores, k = self.top_k, largest=True)
            top_indices = top_indices.tolist()
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            raw_ans = question.golden_answer
            assistant_response = ", ".join(raw_ans) if isinstance(raw_ans, list) else str(raw_ans)
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            })
        with open(self.dev_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.dev_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        topic_indices = {}
        for topic in self.test_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            stored_docs = []
            docs_pass_to_Sbert = []
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_Sbert.append(retrieval_text)   
                         
            embeddings = self.model.encode(docs_pass_to_Sbert, convert_to_tensor=True, show_progress_bar=False)
            

            topic_indices[topic_id] = {
                "docs": stored_docs,
                "embeddings": embeddings
            }
        
        formatted_data = []  
        for question in self.test_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]

            query_embedding = self.model.encode (query_text, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = cos_sim (query_embedding, topic_index ["embeddings"])[0]

            _, top_indices = torch.topk(cosine_scores, k = self.top_k, largest=True)
            top_indices = top_indices.tolist()
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            })
        with open(self.test_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.test_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        
class HybridRetriever(RetrieverBase):
    def __init__(
        self,
        model_name: str,
        alpha: float,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = SentenceTransformer(model_name)
        self.alpha = alpha 
        
    def retrieve(self):
        topic_indices = {}
        for topic in self.train_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            docs_pass_to_BM25 = []
            stored_docs = []
            docs_pass_to_Sbert = []
            
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_BM25.append(simple_tokenize(retrieval_text))
                docs_pass_to_Sbert.append(retrieval_text)   
                         
            bm25 = BM25Okapi(docs_pass_to_BM25)
            embeddings = self.model.encode(docs_pass_to_Sbert, convert_to_tensor=True, show_progress_bar=False)
            

            topic_indices[topic_id] = {
                "bm25": bm25,
                "docs": stored_docs,
                "embeddings": embeddings
            }
        
        formatted_data = []  
        for question in self.train_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            query_tokens = simple_tokenize (query_text)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]
            bm25_scores = topic_index["bm25"].get_scores(query_tokens)
            bm25_norm = self.normalize_scores(bm25_scores)
            query_embedding = self.model.encode (query_text, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = cos_sim (query_embedding, topic_index ["embeddings"])[0].detach().cpu().numpy()
            cosine_mapped = (cosine_scores + 1.0) / 2.0
            cosine_norm = self.normalize_scores(cosine_mapped)
            final_scores = self.alpha * cosine_norm + (1.0 - self.alpha) * bm25_norm
            top_indices = np.argsort(-final_scores)[:self.top_k]
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            raw_ans = question.golden_answer
            assistant_response = ", ".join(raw_ans) if isinstance(raw_ans, list) else str(raw_ans)
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            })
        with open(self.train_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.train_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        topic_indices = {}
        for topic in self.dev_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            docs_pass_to_BM25 = []
            stored_docs = []
            docs_pass_to_Sbert = []
            
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_BM25.append(simple_tokenize(retrieval_text))
                docs_pass_to_Sbert.append(retrieval_text)   
                         
            bm25 = BM25Okapi(docs_pass_to_BM25)
            embeddings = self.model.encode(docs_pass_to_Sbert, convert_to_tensor=True, show_progress_bar=False)
            

            topic_indices[topic_id] = {
                "bm25": bm25,
                "docs": stored_docs,
                "embeddings": embeddings
            }
        
        formatted_data = []  
        for question in self.dev_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            query_tokens = simple_tokenize (query_text)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]
            bm25_scores = topic_index["bm25"].get_scores(query_tokens)
            bm25_norm = self.normalize_scores(bm25_scores)
            query_embedding = self.model.encode (query_text, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = cos_sim (query_embedding, topic_index ["embeddings"])[0].detach().cpu().numpy()
            cosine_mapped = (cosine_scores + 1.0) / 2.0
            cosine_norm = self.normalize_scores(cosine_mapped)
            final_scores = self.alpha * cosine_norm + (1.0 - self.alpha) * bm25_norm
            top_indices = np.argsort(-final_scores)[:self.top_k]
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            raw_ans = question.golden_answer
            assistant_response = ", ".join(raw_ans) if isinstance(raw_ans, list) else str(raw_ans)
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            })
        with open(self.dev_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.dev_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )
        topic_indices = {}
        for topic in self.test_data[0]:
            topic_id = topic.topic_id
            docs = topic.docs
            
            docs_pass_to_BM25 = []
            stored_docs = []
            docs_pass_to_Sbert = []
            
            for doc in docs:
                title = doc.title
                snippet = doc.snippet
                content = doc.content
                retrieval_text = f"{title} {snippet}"
                if self.content_choice == "snippet":
                    text = snippet
                elif self.content_choice == "content":
                    text = content
                else:
                    raise ValueError(f"Invalid content_choice: {self.content_choice}")
                stored = f"Title: {title}\nContent: {text}"
                stored_docs.append(stored)
                docs_pass_to_BM25.append(simple_tokenize(retrieval_text))
                docs_pass_to_Sbert.append(retrieval_text)   
                         
            bm25 = BM25Okapi(docs_pass_to_BM25)
            embeddings = self.model.encode(docs_pass_to_Sbert, convert_to_tensor=True, show_progress_bar=False)
            

            topic_indices[topic_id] = {
                "bm25": bm25,
                "docs": stored_docs,
                "embeddings": embeddings
            }
        
        formatted_data = []  
        for question in self.test_data[1]:
            topic_id = question.topic_id
            target_event = question.target_event
            if self.query_choice == "target_event":
                query_text = target_event
            elif self.query_choice == "option":
                query_text = self.build_retrieval_query_from_options(question)
            query_tokens = simple_tokenize (query_text)
            
            topic_index = topic_indices[topic_id]
            docs = topic_index ["docs"]
            bm25_scores = topic_index["bm25"].get_scores(query_tokens)
            bm25_norm = self.normalize_scores(bm25_scores)
            query_embedding = self.model.encode (query_text, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = cos_sim (query_embedding, topic_index ["embeddings"])[0].detach().cpu().numpy()
            cosine_mapped = (cosine_scores + 1.0) / 2.0
            cosine_norm = self.normalize_scores(cosine_mapped)
            final_scores = self.alpha * cosine_norm + (1.0 - self.alpha) * bm25_norm
            top_indices = np.argsort(-final_scores)[:self.top_k]
            top_docs = [topic_index["docs"][i] for i in top_indices]

            context = "\n\n".join(
                [f"[Doc {i+1}]\n{d}" for i, d in enumerate(top_docs)]
            )
            user_prompt = self.render_prompt(
                self.prompt_template,
                {
                    "context": context,
                    "target": target_event,
                    "option_A": question.option_A,
                    "option_B": question.option_B,
                    "option_C": question.option_C,
                    "option_D": question.option_D,
                }
            )
            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at causal reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            })
        with open(self.test_output_path, "w", encoding="utf-8") as out_f:
            for item in formatted_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"--> XONG! Đã tạo file '{self.test_output_path}' "
            f"với {len(formatted_data)} mẫu."
        )