#!/usr/bin/env python

import pandas as pd
import sys

import tools


def parse():
    try:
        year = int(sys.argv[1])
    except:
        year = 2018
    return year


@tools.chrono
def read(name):
    return pd.read_csv(f"data/{name}.csv", sep="	", encoding="ISO-8859-1")


def main():
    year = parse()
    df = read(f"consommation_{year}")
    # print("\n".join(df.columns))
    print(df.describe()['Consommation'])


if __name__ == "__main__":
    try:
        main()
    finally:
        times = tools.Clock.report()
        message = "{key:<20} (x{value[0]:<5}): {value[1]:.3f} s"
        tools.report("Clock", times, message)
