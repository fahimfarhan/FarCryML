import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


def main():
    transactions = [
        ["bread", "milk"],
        ["bread", "butter", "milk"],
        ["bread", "butter"],
        ["milk", "eggs"],
        ["bread", "milk", "eggs"],
        ["bread", "butter", "milk"],
        ["milk", "butter"],
        ["bread", "butter", "eggs"],
    ]

    # Convert transactions to one-hot format
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)

    df = pd.DataFrame(
        encoded,
        columns=encoder.columns_
    )

    # Find frequent itemsets
    frequent_itemsets = fpgrowth(
        df,
        min_support=0.3,
        use_colnames=True
    )

    print("Frequent itemsets:")
    print(frequent_itemsets)

    # Generate association rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.7
    )

    print("\nAssociation rules:")
    print(
        rules[
            ["antecedents", "consequents",
             "support", "confidence", "lift"]
        ]
    )


if __name__ == "__main__":
    main()