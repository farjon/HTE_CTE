from CTE.bin.HTE_experiments.HTE_Letter_recognition import main as LR_main
import argparse
def main():
    # choose parameter for line search
    # 1 - search optimal number of ferns (will be checked with 1 layers and 7 bit functions)
    # 2 - search optimal number of bit functions (will be checked with 2 layers and 50 ferns)
    # 3 - search optimal number of layers
    search_param = 1
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    if search_param == 1:
        num_of_ferns = [20, 50, 70, 100, 150, 200]
        args.number_of_BF = 7

        for i in range(len(num_of_ferns)):
            args.experiment_number = "line_search_" + str(i)
            args.num_of_ferns = num_of_ferns[i]
            LR_main(args)

    if search_param == 2:
        args.num_of_ferns = 70
        number_of_BF = [5, 6, 7, 8, 9 ,10]

        for i in range(len(number_of_BF)):
            args.experiment_number = "line_search" + str(i)
            args.number_of_BF = number_of_BF[i]
            LR_main(args)






if __name__ == '__main__':
    main()