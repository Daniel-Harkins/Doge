#--------------------------------------------------------------
#Created by Daniel Harkins on 2/18/2025
#This code generates XML from flat files
#Modify the number in row 44 to change the number of outputs
#--------------------------------------------------------------

import xml.etree.ElementTree as ET
import os
from tkinter import Tk, filedialog, messagebox
from faker import Faker
import random


fake = Faker()

def generate_random_person():
    """Generate a random person's data using Faker."""
    first_name = fake.first_name()
    last_name = fake.last_name()
    age = random.randint(18, 110)  # Random age between 18 and 110
    city = fake.city()
    return f"{first_name},{last_name},{age},{city}"

def create_xml_from_flat_file(input_file, output_file):
    # Create the root element
    root = ET.Element("People")

    # Read the flat file
    with open(input_file, 'r') as file:
        for line in file:
            # Split the line into fields (assuming comma-separated values)
            fields = line.strip().split(',')

            # Create a person element
            person = ET.SubElement(root, "Person")

            # Add sub-elements for each field
            ET.SubElement(person, "FirstName").text = fields[0]
            ET.SubElement(person, "LastName").text = fields[1]
            ET.SubElement(person, "Age").text = fields[2]
            ET.SubElement(person, "City").text = fields[3]

    # Add additional fictitious test subjects
    for _ in range(1000):
        random_person = generate_random_person()
        fields = random_person.split(',')

        # Create a person element
        person = ET.SubElement(root, "Person")

        # Add sub-elements for each field
        ET.SubElement(person, "FirstName").text = fields[0]
        ET.SubElement(person, "LastName").text = fields[1]
        ET.SubElement(person, "Age").text = fields[2]
        ET.SubElement(person, "City").text = fields[3]

    # Create an ElementTree object and write to file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def select_input_file():
    # Create a Tkinter root window (hidden)
    root = Tk()
    root.withdraw()  # Hide the root window

    # Show a message box instructing the user to select an input file
    messagebox.showinfo(
        "Select Input File",
        "Please select the input file (e.g., a .txt file) to generate the XML."
    )

    # Open a file dialog to select the input file
    input_file = filedialog.askopenfilename(
        title="Select Input File",
        filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
    )

    return input_file

if __name__ == "__main__":
    # Let the user select the input file
    input_file = select_input_file()

    if input_file:
        # Define the output file name
        output_file = "output.xml"

        # Generate the XML file
        create_xml_from_flat_file(input_file, output_file)

        # Get the absolute path of the output file
        output_file_path = os.path.abspath(output_file)

        # Print the full file location
        print(f"XML file has been generated at: {output_file_path}")
    else:
        print("No input file selected. Exiting.")
