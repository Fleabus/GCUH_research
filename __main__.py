from setup import setup_data

if __name__ == "__main__":
    data_type = input("What lead? (e.g. MLII, V1, V2...):\n")
    data_formatter = setup_data(data_type.upper())
    print(data_formatter.x)
