"""
Quick test for DataParallelTransformer
"""
import torch
import torch.nn as nn

def quick_test():
    print("🧪 Quick DataParallel Test")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.device_count() < 2:
        print("❌ Need at least 2 GPUs")
        return
    
    # Simple test model
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 256)
            self.linear = nn.Linear(256, 1000)
            
        def encode(self, x, mask=None):
            return self.embed(x)
        
        def decode(self, enc_out, src_mask, tgt, tgt_mask=None):
            return self.embed(tgt) + enc_out
        
        def project(self, x):
            return self.linear(x)
    
    # Import our custom class
    from train import DataParallelTransformer
    
    # Create model
    model = SimpleTransformer().cuda(0)
    model_dp = DataParallelTransformer(model, device_ids=[0, 1])
    
    print("✅ Model wrapped successfully")
    
    # Test data
    batch_size = 8
    seq_len = 10
    
    enc_input = torch.randint(0, 1000, (batch_size, seq_len)).cuda(0)
    dec_input = torch.randint(0, 1000, (batch_size, seq_len)).cuda(0)
    enc_mask = torch.ones(batch_size, seq_len).cuda(0)
    dec_mask = torch.ones(batch_size, seq_len).cuda(0)
    
    print(f"Input shapes: {enc_input.shape}")
    
    # Clear memory
    torch.cuda.empty_cache()
    
    try:
        # Test forward_training
        print("Testing forward_training...")
        with torch.no_grad():
            output = model_dp.forward_training(enc_input, enc_mask, dec_input, dec_mask)
        
        print(f"✅ Output shape: {output.shape}")
        
        # Check GPU memory
        mem0 = torch.cuda.memory_allocated(0) / 1024**3
        mem1 = torch.cuda.memory_allocated(1) / 1024**3
        
        print(f"GPU Memory - GPU 0: {mem0:.2f}GB, GPU 1: {mem1:.2f}GB")
        
        if mem1 > 0.01:  # At least 10MB on GPU 1
            print("🎉 SUCCESS! Both GPUs are being used!")
        else:
            print("❌ Only GPU 0 is being used")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test() 