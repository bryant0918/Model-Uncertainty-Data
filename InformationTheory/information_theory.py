"""
Information Theory Lab

Bryant McArthur
Section
9/20/22
"""

import numpy as np
import wordle


# Problem 1
def get_guess_result(true_word, guess):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the true word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        true_word (string) - the secret word
        guess (string) - the guess being made
    Returns:
        result (array of integers) - the result of the guess, as described above
    """
    # set initial result and create list of guess by character
    result = np.array([0, 0, 0, 0, 0])
    new_true = [x for x in true_word]
    new_guess = [x for x in guess]

    # Go through the letters and append 2 if it's exact
    for i, letter in enumerate(guess):
        if letter == true_word[i]:
            result[i] = 2
            new_guess[i] = None
            new_true[i] = None

    # Go through the remaining letters
    for i, letter in enumerate(new_guess):
        if letter is None:  # Pass if already equal to 2
            continue
        if letter in new_true[:] and result[i] != 2:
            result[i] = 1
            new_guess[i] = None
            if letter in new_true and letter in new_guess: # Account for duplicates
                for j, l in enumerate(new_guess):
                    if letter == l:
                        new_guess[j] = None

    return result




# Problem 2
def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they 
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words




def get_all_guess_results(possible_words, allowed_words):
    """
    Calculates the result of making every guess for every possible secret word
    
    Arguments:
        possible_words (list of strings)
            A list of all possible secret words
        allowed_words (list of strings)
            A list of all allowed guesses
    Returns:
        ((n,m,5) ndarray) - the results of each guess for each secret word,
            where n is the the number of allowed guesses and m is number of possible secret words.
    """
    # Initialize jarray
    my_array = np.zeros((len(allowed_words), len(possible_words), 5))

    # Go through the allowed words and possible words
    for i, a_word in enumerate(allowed_words):
        for j, p_word in enumerate(possible_words):
            result = get_guess_result(p_word, a_word)
            my_array[i][j] = result

    return my_array





# Problem 3
def compute_highest_entropy(all_guess_results, allowed_words):
    """
    Compute the entropy of each guess.
    
    Arguments:
        all_guess_results ((n,m,5) ndarray) - the output of the function
            from Problem 2, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_words (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
        (int) Index of the highest-entropy guess
    """
    all_results = all_guess_results

    # For array comprehension
    base_three = np.array([3**i for i in range(5)])
    base3 = np.sum(all_results*base_three, axis=2)

    # Use the np.unique function in the spec file
    counts = np.array([np.unique(row, return_counts=True)[1] for row in base3])
    denom = 2309

    # Calculate the entropy
    entropy = [np.sum([-(num/denom)*np.log2(num/denom) for num in row]) for row in counts]

    return allowed_words[np.argmax(entropy)], np.argmax(entropy)


# Problem 4
def filter_words(all_guess_results, possible_words, guess_idx, result):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already computed the result of all guesses for all possible words in 
    Problem 2, we will use this array instead of recomputing the results.
    
	Return a filtered list of possible words that are still possible after 
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array will be used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (3-D ndarray)
            The output of Problem 2, containing the result of making
            any allowed guess for any possible secret word
        possible_words (list of str)
            The list of possible secret words
        guess_idx (int)
            The index of the guess that was made in the list of allowed guesses.
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (3-D ndarray) The filtered array of guess results
    """
    # Create a mask and then filter on the right axis
    mask = np.all(all_guess_results[guess_idx] == result, axis=1)
    filtered = all_guess_results[:, mask, :]
    return np.array(possible_words)[mask], filtered


# Problem 5
def play_game_naive(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)

    # Play as long as the game isn't done
    while len(possible_words) > 1:
        guess_idx = np.random.choice(len(allowed_words)) # Make a random choice
        result, count = game.make_guess(allowed_words[guess_idx])
        possible_words, all_guess_results = filter_words(all_guess_results, possible_words, guess_idx, result)

    return game.guess_ct


# Problem 6
def play_game_entropy(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)

    # As long as there are more than one possible words we haven't won
    while len(possible_words) > 1:
        guess, index = compute_highest_entropy(all_guess_results, allowed_words) # Compute entropy
        result, count = game.make_guess(guess)                                  # Guess then filter
        possible_words, all_guess_results = filter_words(all_guess_results, possible_words, index, result)

    return game.guess_ct


# Problem 7
def compare_algorithms(all_guess_results, possible_words, allowed_words, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.
    
    
    Arguments:
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    # Define my game and my count lists
    game = wordle.WordleGame("allowed_words.txt", "possible_words.txt")
    naive = []
    entropy = []

    # Play the games n times and append the count.
    for i in range(n):
        entropy.append(play_game_entropy(game, all_guess_results, possible_words, allowed_words, None))
        naive.append(play_game_naive(game, all_guess_results, possible_words, allowed_words, None))

    return np.mean(naive), np.mean(entropy)


if __name__ == "__main__":
    # Problem 1
    print(get_guess_result("ennnn", "steel"))
    print(get_guess_result("zebra", "aahed"))
    print(get_guess_result("nnnen", "steel"))
    print(get_guess_result("boxed", "excel"))

    # Problem 2
    possible_words = load_words("possible_words.txt")
    allowed_words = load_words("allowed_words.txt")
    print(get_all_guess_results(possible_words[:3], allowed_words[:2]))
    all_results = get_all_guess_results(possible_words, allowed_words)
    np.save("C:/Users/bryan/Documents/School/Fall 2022/Math 403/volume3/InformationTheory/all_results.npy", all_results)