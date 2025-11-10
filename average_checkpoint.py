import torch
import argparse
import os
import glob
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser('Checkpoint Averaging')

    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing model checkpoints to average.')
    parser.add_argument('--number-of-checkpoints', type=int, default=5,
                        help='Number of last checkpoints to average.')
    parser.add_argument('--output-checkpoint', type=str, required=True,
                        help='Path to save the averaged checkpoint.')
    parser.add_argument('--pattern', type=str, default='checkpoint_epoch_*.pt',
                        help='Pattern to match checkpoint files.')
    

    args = parser.parse_args()
    return args

def find_latest_checkpoints(checkpoint_dir, pattern, num_checkpoints):
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    all_checkpoints.sort()

    print(f"\nFound {len(all_checkpoints)} checkpoints total")
    print(f"Using last {min(num_checkpoints, len(all_checkpoints))} for averaging:")
    
    latest = all_checkpoints[-num_checkpoints:]

    for f in latest:
        print(f"  - {os.path.basename(f)}")
    
    return latest

def load_checkpoint(filepath):
    return torch.load(filepath, map_location='cpu')

def average_checkpoints(checkpoint_files):

    # Load the first checkpoint to get the structure
    first_checkpoint = load_checkpoint(checkpoint_files[0])

    # Initialize an empty state dict for averaging
    averaged_state_dict = OrderedDict()

    # Prepare the keys
    for key, value in first_checkpoint['model'].items():
        averaged_state_dict[key] = torch.zeros_like(value)

    print(f"Averaging {len(checkpoint_files)} checkpoints...")

    # Sum the parameters
    for i, filepath in enumerate(checkpoint_files, start=1):
        checkpoint = load_checkpoint(filepath)
        state_dict = checkpoint['model']
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] += state_dict[key]
        print(f"  Processed {i}/{len(checkpoint_files)}: {filepath}")
    
    # compute average
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] /= len(checkpoint_files)

    # Create the final checkpoint
    final_checkpoint = {
        'model': averaged_state_dict,
        'args': first_checkpoint.get('args'),
        'epoch': first_checkpoint.get('epoch'),
        'best_loss': first_checkpoint.get('best_loss'), 
        'last_epoch': first_checkpoint.get('last_epoch'),
        'val_loss': first_checkpoint.get('val_loss'),
    }
    return final_checkpoint

def main():
    args = get_args()
    checkpoint_files = find_latest_checkpoints(args.checkpoint_dir, args.pattern, args.number_of_checkpoints)
    if len(checkpoint_files) == 0:
        print("No checkpoints found to average.")
        return
    if len(checkpoint_files) < args.number_of_checkpoints:
        print(f"Warning: Found only {len(checkpoint_files)} checkpoints, less than the requested {args.number_of_checkpoints}.")

    averaged_checkpoint = average_checkpoints(checkpoint_files)
    torch.save(averaged_checkpoint, args.output_checkpoint)
    print(f"Averaged checkpoint saved to {args.output_checkpoint}")
    
if __name__ == "__main__":
    main()


