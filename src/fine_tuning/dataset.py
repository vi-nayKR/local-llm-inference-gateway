from typing import List, Dict

class InstructionDataset:
    """
    Domain instruction dataset builder for Small Language Models (SLMs).
    Formats samples into standard ChatML and Llama 3.2 prompt templates.
    """
    @staticmethod
    def get_sample_compliance_data() -> List[Dict[str, str]]:
        return [
            {
                "instruction": "Evaluate the risk level of the following digital asset transaction.",
                "input": "Transfer of $250,000 USDT to an unverified external address with 0 prior transaction history within a 1-hour window.",
                "output": "Risk Level: CRITICAL (Score: 0.96). Action Required: Trigger multi-signature quorum consensus (M-of-N) threshold rule and place 24-hour velocity hold pending AML compliance review."
            },
            {
                "instruction": "Identify policy violations in the proposed wallet mutation request.",
                "input": "Admin user attempting to add destination address '0x742d...44e' to whitelist without 2FA supervisor sign-off.",
                "output": "Policy Violation Detected: E-WH-402 (Missing Dual-Key Authorization). Mutation rejected. System requires dual supervisor signature for whitelist alterations."
            },
            {
                "instruction": "Summarize the key architectural benefits of PagedAttention in vLLM.",
                "input": "",
                "output": "PagedAttention treats Key-Value (KV) cache memory as non-contiguous physical pages (similar to OS virtual memory), reducing VRAM memory fragmentation from ~60-80% down to under 4% and boosting batch serving throughput."
            }
        ]

    @staticmethod
    def format_llama3(sample: Dict[str, str]) -> str:
        """Formats an instruction-input-output triplet into the Llama 3.2 Chat template."""
        system_msg = "<|start_header_id|>system<|end_header_id|>\nYou are an enterprise AI compliance and systems engineering assistant.<|eot_id|>"
        user_content = sample["instruction"]
        if sample.get("input"):
            user_content += f"\n\nContext:\n{sample['input']}"
        user_msg = f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>"
        assistant_msg = f"<|start_header_id|>assistant<|end_header_id|>\n{sample['output']}<|eot_id|>"
        
        return f"<|begin_of_text|>{system_msg}\n{user_msg}\n{assistant_msg}"

dataset_builder = InstructionDataset()
