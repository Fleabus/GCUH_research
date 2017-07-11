from setup import setup_data

if __name__ == "__main__":
    data_type = input("What lead? (e.g. MLII, V1, V2...):\n")
    data_formatter = setup_data(data_type.upper())
    data_formatter.counter()
    data_formatter.shuffle()
    data_formatter.counter()
    data_formatter.equalize_data()
    data_formatter.counter()
