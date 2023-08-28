
# TODO: COMBINE THESE 
def validate_option_type(option_type):
    """Validates that a passed option_type is one of 'call' or 'put' """
    if option_type not in ("call", "put"):
        raise ValueError("Invalid option_type. Allowed values are 'call' or 'put'.")
    
def validate_d_i(i):
    """Validates that a passed option_type is one of 'call' or 'put' """
    if i not in (1, 2):
        raise ValueError("Invalid option_type. Allowed values are 1 or 2.")