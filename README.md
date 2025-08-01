#  Insurance Policy Query API (Falcon-7B + LlamaIndex)

This project enables intelligent querying of insurance policy documents using the **Falcon-7B-Instruct** large language model and `llama-index`. It was developed for **HackRx 6.0** to assist users in getting structured, clause-based answers from unstructured PDF documents such as health insurance policies.

---

##  Features

-  Natural language question answering over PDFs
-  Uses open-source **Falcon-7B-Instruct** model
-  Cites specific clauses from policy documents
-  Supports multiple documents
-  Query response in **JSON** (decision + reason + reference)

---

##  Sample Queries

```json
{
  "query": "A 46-year-old male had knee surgery in Pune. His policy is 3 months old. Is he eligible for a claim?"
}
