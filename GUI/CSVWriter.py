import io
import csv

class CSVWriter:
    """
    A CSV writer which will write rows to CSV file
    """
    def __init__(self, filename, column_headers):
        # Redirect output to a queue
        self.filename = filename
        self.writerow(column_headers, write_type='w')
        
    def writerow(self, data, write_type='a'):
        # Output to csv file
        with open(self.filename, write_type, newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',') 
            writer.writerow(data)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)