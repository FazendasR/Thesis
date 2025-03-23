#imports 
import sys

# Below are a few examples of how you can build your required functions

def data_preparation(path_out):
    """Function description
    Args:
        - X (TYPE): argument description
    Returns:
        - Y (TYPE): returned value description
    """    
    pass
    
    
def train(path_in, ml_uri, ml_certificate):
    """Function description
    Args:
        - X (TYPE): argument description
    Returns:
        - Y (TYPE): returned value description
    """
    pass
    
    
def predict(path_out, path_in, ml_uri, ml_certificate):
    """Function description
    Args:
        - X (TYPE): argument description
    Returns:
        - Y (TYPE): returned value description
    """
    pass

    
def monitor(path_in):
    """Function description
    Args:
        - X (TYPE): argument description
    Returns:
        - Y (TYPE): returned value description
    """
    pass

    
def main():
    args = sys.argv[1:]
    
    if len(args) == 2 and args[0] == '-data-preparation':
        print(data_preparation(args[1]))

    if len(args) == 4 and args[0] == '-train':
        print(train(args[2], args[3], args[4]))
    
    if len(args) == 5 and args[0] == '-predict':
        print(predict(args[1], args[2], args[3], args[4]))

    if len(args) == 2 and args[0] == '-monitor':
        print(predict(args[1]))


if __name__ == "__main__":
    main()   