import argparse
parser = argparse.ArgumentParser(description='Script to export train, validation and test data')

parser.add_argument('-i', '--i', help='input string: names of input data folders', required=True)
parser.add_argument('-o', '--o', help='output string', required=True)
args = parser.parse_args()
print(args.i, args.o)

print("Input strings: %s" % args.i)
print("Output string: %s"% args.o)
