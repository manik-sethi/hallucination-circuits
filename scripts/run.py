# scripts/run.py
import io
import argparse
from sae_bench import encode


def main():

    parser = argparse.ArgumentParser("SAE similarity study CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("encode", help="Encode dataset into SAE representations")
    p.add_argument("--model", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True)


    args = parser.parse_args()
    
    if args.cmd == "encode":
        encode.run(io.combined_path(), args.model, args.layers)

if __name__ == "__main__":
    main()