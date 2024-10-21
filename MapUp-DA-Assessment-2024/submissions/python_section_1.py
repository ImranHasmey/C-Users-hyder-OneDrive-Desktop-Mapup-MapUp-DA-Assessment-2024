#Question 1 
from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    reverse_lt = []
    length = len(lst)

    for i in range (0,length,n):
        group=[]
        
        for j in range(i,min(i+n,length)):
            group.append(lst[j])
            
        for k in range(len(group)-1,-1,-1):
            reverse_lt.append(group[k])
   
    return reverse_lt

print(reverse_by_n_elements([1,2,3,4,5,6,7,8],3))
print(reverse_by_n_elements([1,2,3,4,5],2))
print(reverse_by_n_elements([10,20,30,40,50,60,70],4))
    
#Question 2
def group_strings_by_length(strings: List[str]) -> Dict[int, List[str]]:
        length_dict = {} 
        
        for string in strings:
            length = len(string)  
            
            if length not in length_dict:
                length_dict[length] = []
            
            length_dict[length].append(string)
        
        return dict(sorted(length_dict.items()))
    
print(group_strings_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
print(group_strings_by_length(["one", "two", "three", "four"]))  
    
    
#Question 3
from typing import Dict, Any

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    flat_dict = {}

    def flatten(current_dict: Dict[str, Any], parent_key: str = ''):
        for key, value in current_dict.items():
            # Create the new key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
               
                flatten(value, new_key)
            elif isinstance(value, list):
               
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        flatten(item, f"{new_key}[{index}]")
                    else:
                        flat_dict[f"{new_key}[{index}]"] = item
            else:
               
                flat_dict[new_key] = value

    flatten(nested_dict)
    return flat_dict   


nested_input = {"road": {"name": "Highway 1","length": 350,
                         "sections": [{"id": 1,
                        "condition": {"pavement": "good",
                        "traffic": "moderate"
                        }
                        }]
                        }}

flattened_output = flatten_dict(nested_input)
print(flattened_output)
   
#Question 4
from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(path: List[int], used: List[bool]):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False

    nums.sort()
    result = []
    backtrack([], [False] * len(nums))
    return result


print(unique_permutations([1, 1, 2]))

#Question 5
import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    Find and return all valid dates from the input text in specified formats.

    :param text: The input string containing dates in various formats.
    :return: A list of valid dates found in the text.
    """
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b' ]
    
    combined_pattern = '|'.join(patterns)
    
    matches = re.findall(combined_pattern, text)
    return matches

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))

#Question 6
# first install polyline #pip install polyline
import pandas as pd
import polyline
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000  
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]  
    for i in range(1, len(coordinates)):
        dist = haversine(df.latitude[i-1], df.longitude[i-1], df.latitude[i], df.longitude[i])
        distances.append(dist)
    
    df['distance'] = distances
    return df

polyline_str = "_p~iF~wmy@_@v_zC|b@_q@{n@zI"
df = polyline_to_dataframe(polyline_str)
print(df)

#Question 7
from typing import List

def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    rotated = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]
    
    return rotated

def transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    transformed = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(matrix[i])  
            col_sum = sum(matrix[k][j] for k in range(n))  
            transformed[i][j] = row_sum + col_sum - matrix[i][j] 

    return transformed

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    rotated = rotate_matrix(matrix)  
    final_matrix = transform_matrix(rotated)  
    return final_matrix

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
for row in result:
    print(row)

#Question 8
import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    grouped = df.groupby(['id', 'id_2'])

    results = pd.Series(index=grouped.groups.keys(), dtype=bool)

    for (id_value, id_2_value), group in grouped:
        start_times = group['start']
        end_times = group['end']
        
        full_week_covered = (start_times.dt.dayofweek.min() == 0 and start_times.dt.dayofweek.max() == 6)
        full_day_covered = (start_times.min().time() == pd.Timestamp('00:00:00').time() and
                            end_times.max().time() == pd.Timestamp('23:59:59').time())

      
        results[(id_value, id_2_value)] = not (full_week_covered and full_day_covered)

    return results


# Sample DataFrame
data = {
        'id': [1, 1, 2, 2],
        'id_2': [1, 1, 1, 1],
        'startDay': ['2024-10-14', '2024-10-15', '2024-10-14', '2024-10-20'],
        'startTime': ['00:00:00', '00:00:00', '00:00:00', '00:00:00'],
        'endDay': ['2024-10-14', '2024-10-15', '2024-10-14', '2024-10-20'],
        'endTime': ['23:59:59', '23:59:59', '23:59:59', '23:59:59'],
    }
    
df = pd.DataFrame(data)
result = time_check(df)
print(result)
