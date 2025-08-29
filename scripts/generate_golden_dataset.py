import json
from golden_data import DATA

def create_golden_dataset():
    """Crea el archivo evaluation_dataset.jsonl desde la fuente de datos."""
    
    with open("evaluation_dataset.jsonl", 'w', encoding='utf-8') as f:
        for entry in DATA:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("El archivo evaluation_dataset.jsonl ha sido creado exitosamente.")

if __name__ == "__main__":
    create_golden_dataset()