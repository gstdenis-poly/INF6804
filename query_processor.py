import sys

# Program main function
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('ERROR: Wrong arguments (see -h)')
    elif sys.argv[1] == '-h':
        help += 'TODO'
        print(help)
    else:
        print('TODO')