"""Volume 3: Web Scraping.
Bryant McArthur
Math 405
January 17, 2023
"""

import requests
from bs4 import BeautifulSoup
from os import path
from matplotlib import pyplot as plt


# Problem 1
def prob1():
    """Use the requests library to get the HTML source for the website 
    http://www.example.com.
    Save the source as a file called example.html.
    If the file already exists, do not scrape the website or overwrite the file.
    """

    # Check if path exists
    if not path.exists("example.html"):
        # Scrape site and save file
        response = requests.get("http://www.example.com")
        with open("example.html", 'w') as f:
            f.write(response.text)
    pass


# Problem 2
def prob2(code):
    """Return a list of the names of the tags in the given HTML code.
    Parameters:
        code (str): A string of html code
    Returns:
        (list): Names of all tags in the given code"""
    # parse and get tags
    soup = BeautifulSoup(code, 'html.parser')
    tags = soup.find_all(True)

    # Iterate through tags and append names
    names = []
    for tag in tags:
        names.append(tag.name)

    return names


# Problem 3
def prob3(filename="example.html"):
    """Read the specified file and load it into BeautifulSoup. Return the
    text of the first <a> tag and whether or not it has an href
    attribute.
    Parameters:
        filename (str): Filename to open
    Returns:
        (str): text of first <a> tag
        (bool): whether or not the tag has an 'href' attribute
    """
    # Open file and parse
    with open(filename, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Get first a tag
    a_tag = soup.a

    # Check if it has href attribute
    my_bool = hasattr(a_tag, 'href')

    return a_tag.text, my_bool


# Problem 4
def prob4(filename="san_diego_weather.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the following tags:

    1. The tag containing the date 'Thursday, January 1, 2015'.
    2. The tags which contain the links 'Previous Day' and 'Next Day'.
    3. The tag which contains the number associated with the Actual Max
        Temperature.

    Returns:
        (list) A list of bs4.element.Tag objects (NOT text).
    """
    # Open file and parse
    with open(filename, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Find tags
    one = soup.find(string="Thursday, January 1, 2015").parent
    twoa = soup.find_all(class_="previous-link")[0]
    twob = soup.find_all(class_="next-link")[0]
    three = soup.find(string="Max Temperature").parent.parent.next_sibling.next_sibling.span.span

    return [one, twoa, twob, three]


# Problem 5
def prob5(filename="large_banks_index.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the tags containing the links to bank data from September 30, 2003 to
    December 31, 2014, where the dates are in reverse chronological order.

    Returns:
        (list): A list of bs4.element.Tag objects (NOT text).
    """
    # Open file and parse
    with open(filename, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Find all tags
    tags = soup.find_all(name='a')

    # Define first and last tags
    first = soup.find(string="December 31, 2014").parent
    last = soup.find(string="September 30, 2003").parent

    # Iterate through appropriate tags and save
    append = False
    mylist = []
    for tag in tags:
        if tag == first:
            append = True
        if append:
            if hasattr(tag, 'href'):
                mylist.append(tag)
        if tag == last:
            break

    return mylist


# Problem 6
def prob6(filename="large_banks_data.html"):
    """Read the specified file and load it into BeautifulSoup. Create a single
    figure with two subplots:

    1. A sorted bar chart of the seven banks with the most domestic branches.
    2. A sorted bar chart of the seven banks with the most foreign branches.

    In the case of a tie, sort the banks alphabetically by name.
    """
    # Open file and parse
    with open(filename, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Iterate through tables and get appropriate rows
    bank_tags = [tag.td.text[:tag.td.text.find("/")] for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]
    bank_ids = [tag.td.next_sibling.next_sibling.next_sibling.next_sibling.text
                for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]
    domestic_branches = [tag.td.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.text
                         for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]
    foreign_branches = [tag.td.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.text
                         for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]

    # Change bad values to 0
    for i, bank in enumerate(domestic_branches):
        if bank == ".":
            domestic_branches[i] = "0"
            foreign_branches[i] = "0"

    # Create appropriate dictionaries
    domestic_banks = {(bank, bank_ids[i]): float(domestic_branches[i].replace(",", "")) for i, bank in enumerate(bank_tags)}
    foreign_banks = {(bank, bank_ids[i]): float(foreign_branches[i].replace(",", "")) for i, bank in enumerate(bank_tags)}

    # Sort
    sorted_domestic = sorted(domestic_banks.items(), key=lambda item: item[1], reverse=True)
    sorted_foreign = sorted(foreign_banks.items(), key=lambda item: item[1], reverse=True)

    # First Plot
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].barh([x[0][0] for x in sorted_domestic[:7]], [x[1] for x in sorted_domestic[:7]])

    # Second Plot
    axes[1].barh([x[0][0] for x in sorted_foreign[:7]], [x[1] for x in sorted_foreign[:7]])
    plt.title("Foreign Branches")
    plt.suptitle("Banks with Most Domestic Branches")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(prob4())
