import torch
def check_torch():
    print("Verificación del entorno PyTorch\n")

    print(f"Versión de torch: {torch.__version__}")
    if torch.cuda.is_available():
        print("✅ CUDA está disponible.")
        print(f"🖥️ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"🧠 Memoria total: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB")
        print(f"⚙️ Núcleos CUDA: {torch.cuda.get_device_properties(0).multi_processor_count}")
    else:
        print("⚠️ CUDA NO está disponible.")
        print("🔧 Se usará CPU. Si tienes GPU NVIDIA, instala torch con CUDA:\n")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")