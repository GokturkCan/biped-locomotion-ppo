import json
from biped_locomotion.smoke import run

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
