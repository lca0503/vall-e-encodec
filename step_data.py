import csv
from argparse import ArgumentParser

import jsonlines


def deduplicate(origin_code):
    deduplicate_code = [origin_code[0]]
    for code in origin_code:
        if code != deduplicate_code[-1]:
            deduplicate_code.append(code)
    return deduplicate_code

    
def main(args):
    with jsonlines.open(args.input_file) as reader:
        with open(args.output_file, 'w') as output_fp:
            writer = csv.writer(output_fp)
            for obj in reader:
                # Get and deduplicate 3 secs of codes
                source_code = obj['code'][:150]
                source_code = deduplicate(source_code)
                source_code_str = "".join(f"v_tok_{tok}" for tok in source_code)

                target_code = obj['merged_code']
                assert target_code[:len(source_code)] == source_code
                # All codes
                target_code_str = "".join(f"v_tok_{tok}" for tok in target_code)

                if args.step == 1:
                    writer.writerow([source_code_str, target_code_str])
                else:
                    source_text = obj['text']
                    source = source_text + "</s>" + source_code_str
                    writer.writerow([source, target_code_str])
    
                    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default='superb_asr_validation_chunk_30_mhubert_layer11_code1000_norm_False_beam_False_topk_3_beamsize_1.jsonl',
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default='validation.csv',
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=2,
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
