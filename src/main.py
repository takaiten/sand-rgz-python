from pathlib import Path
from sys import argv
from numpy import log, log2, sqrt
from scipy.stats import t, chi2
import csv

N = 909768
alpha = 0.0005
asFirstWord = []
asSecondWord = []
template = [
    ('J', 'N'),
    ('N', 'N'),
    ('V', 'I'),
    ('V', 'N'),
]


def read_csv(path, word, part_of_speech=None):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            # if first word in bigram is our word
            if row[1] == word and row[2] == part_of_speech:
                # save it to array of bigrams with word in first place
                asFirstWord.append(row)
            # if second word in bigram is our word
            elif row[3] == word and row[4] == part_of_speech:
                # save it to array of bigrams with word in second place
                asSecondWord.append(row)


def write_csv(directory, filename, data, quantile=None):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(directory + filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        # Skip one line for header
        writer.writerow([])
        # Write array of data
        writer.writerows(data)
        # If quantile was provided write it as last row
        if quantile:
            writer.writerow([quantile])


def exclude_by_template(data):
    result = []
    for row in data:
        for variant in template:
            if variant[0] == row[2] and variant[1] == row[4]:
                result.append(row)
    return result


def calculate_t_test(data):
    result = []
    for row in data:
        o11, o12, o21, o22 = [int(x) for x in row[5:]]

        c1 = o11 + o12
        c2 = o11 + o21
        x_m = o11 / N
        mu = c1 * c2 / (N ** 2)
        t = (x_m - mu) / sqrt((x_m * (1 - x_m)) / N)

        result.append([row[0], row[1], row[3], t])
    return sorted(result, key=lambda x: x[3], reverse=True)


def calculate_chi_test(data):
    result = []
    for row in data:
        o11, o12, o21, o22 = [int(x) for x in row[5:]]

        chi_square = (N * (o11 * o22 - o12 * o21) ** 2) / ((o11 + o12) * (o11 + o21) * (o12 + o22) * (o21 + o22))

        result.append([row[0], row[1], row[3], chi_square])
    return sorted(result, key=lambda x: x[3], reverse=True)


def logL(a, b, c):
    return a * log(c) + (b - a) * log(1 - c)


def calculate_likelihood_ratio_test(data):
    result = []
    for row in data:
        o11, o12, o21, o22 = [int(x) for x in row[5:]]

        c12 = o11
        c1 = o11 + o12
        c2 = o11 + o21

        # likelihood ratio
        lr = -2 * (
                + logL(c12, c1, c2 / N)
                + logL(c2 - c12, N - c1, c2 / N)
                - logL(c12, c1, c12 / c1)
                - logL(c2 - c12, N - c1, (c2 - c12) / (N - c1))
        )

        result.append([row[0], row[1], row[3], lr])
    return sorted(result, key=lambda x: x[3], reverse=True)


def calculate_point_mutual_information(data):
    result = []
    for row in data:
        o11, o12, o21, o22 = [int(x) for x in row[5:]]

        c12 = o11
        c1 = o11 + o12
        c2 = o11 + o21

        pmi = log2((N * c12) / (c1 * c2))

        result.append([row[0], row[1], row[3], pmi])
    return sorted(result, key=lambda x: x[3], reverse=True)


def calculate_mutual_information(data):
    result = []
    for row in data:
        o11, o12, o21, o22 = [int(x) for x in row[5:]]

        c1 = o11 + o12
        c2 = o11 + o21

        mi = (o11 / N) * log2((N * o11) / (c1 * c2)) + \
             (o12 / N) * log2((N * o12) / (c1 * (o22 + o12))) + \
             (o21 / N) * log2((N * o21) / ((N - c1) * c2)) + \
             (o22 / N) * log2((N * o22) / ((N - c1) * (N - c2)))

        result.append([row[0], row[1], row[3], mi])
    return sorted(result, key=lambda x: x[3], reverse=True)


def hypothesis_t_test(data):
    quantile = t.ppf(1 - alpha / 2, N - 1)
    return [x for x in data if x[3] > quantile], quantile


def test_hypothesis_w_chi_sqaure(data):
    quantile = chi2.ppf(1 - alpha, 1)
    return [x for x in data if x[3] > quantile], quantile


def main():
    if len(argv) != 4:
        print('usage: main.py [path to bigrams csv file] [word] [N|V|J|I]')
        exit(1)

    directory = './output_' + argv[2]

    ''' Point 1 '''

    # Read CSV file and save results in two arrays: asFirstWord, asSecondWord
    read_csv(argv[1], argv[2], argv[3])

    print('Len with first word', len(asFirstWord))
    print('Len with second word', len(asSecondWord))

    # Save first and seconds arrays as CSV files
    write_csv(directory, '/as_first_word.csv', asFirstWord)
    write_csv(directory, '/as_second_word.csv', asSecondWord)

    # Combine both arrays and sort by bigram frequency
    combined = asFirstWord + asSecondWord
    combined = sorted(combined, key=lambda x: int(x[5]), reverse=True)

    print('Len combined', len(combined))

    # Save result array as CSV file
    write_csv(directory, '/combined_n_sorted.csv', combined)

    ''' Point 2 '''

    # Exclude rows by template
    excluded = exclude_by_template(combined)

    # Save excluded by template result array as CSV file
    write_csv(directory, '/excluded.csv', excluded)

    ''' Point 3 '''
    # Calculate t-test
    t_test_data = calculate_t_test(excluded)

    # Save t-test result array as CSV file
    write_csv(directory, '/data_t_test.csv', t_test_data)

    # Calculate chi square
    chi_square_data = calculate_chi_test(excluded)

    # Save chi square result array as CSV file
    write_csv(directory, '/data_chi2.csv', chi_square_data)

    # Calculate likelihood ratio
    lr_data = calculate_likelihood_ratio_test(excluded)

    # Save likelihood ratio result array as CSV file
    write_csv(directory, '/data_lr.csv', lr_data)

    # Calculate point mutual information
    pmi_data = calculate_point_mutual_information(excluded)

    # Save point mutual information result array as CSV file
    write_csv(directory, '/data_pmi.csv', pmi_data)

    # Calculate mutual information
    mi_data = calculate_mutual_information(excluded)

    # Save mutual information result array as CSV file
    write_csv(directory, '/data_mi.csv', mi_data)

    ''' Point 4 '''
    # Filter t-test array of data by Students Quantile with (N-1) DOF and (1 - alpha/2)
    t_test_filtered, quantile = hypothesis_t_test(t_test_data)

    write_csv(directory, '/filtered_t_test.csv', t_test_filtered, quantile)

    # Filter chi square array of data by Chi Square Quantile with (1) DOF and (1 - alpha)
    chi_square_filtered, quantile = test_hypothesis_w_chi_sqaure(chi_square_data)

    write_csv(directory, '/filtered_chi2.csv', chi_square_filtered, quantile)

    # Filter lr array of data by Chi Square Quantile with (1) DOF and (1 - alpha)
    lr_filtered, quantile = test_hypothesis_w_chi_sqaure(lr_data)

    write_csv(directory, '/filtered_lr.csv', lr_filtered, quantile)


if __name__ == '__main__':
    main()
