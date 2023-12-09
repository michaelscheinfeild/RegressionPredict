import colorcet as cc
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import re
from datetime import timedelta
import numpy as np
import mplcursors
from pprint import pprint

MIN_TIME = 5.0
MAX_TIME = 12.0


# plot schedule by anesthetist_id and room improved
def plot_day_scheduleDay(schedule):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(w=2.5 * 9.5, h=2 * 5)
    fig.tight_layout(pad=1.7)

    resources = set(schedule['anesthetist_id'])
    resources = sorted(resources, key=lambda x: (len(x), x), reverse=True)
    resource_mapping = {resource: i for i, resource in enumerate(resources)}

    # Convert start_time and end_time to datetime objects
    schedule['start_time'] = pd.to_datetime(schedule['start_time'])
    schedule['end_time'] = pd.to_datetime(schedule['end_time'])

    # Calculate time intervals in hours from the start of the day
    intervals_start = (schedule['start_time'] - schedule['start_time'].dt.floor('d')).dt.total_seconds().div(3600)
    intervals_end = (schedule['end_time'] - schedule['start_time'].dt.floor('d')).dt.total_seconds().div(3600)

    intervals = list(zip(intervals_start, intervals_end))

    palette = sns.color_palette("husl", n_colors=len(schedule))  # Use a color palette suitable for your data
    palette = [(color[0] * 0.9, color[1] * 0.9, color[2] * 0.9) for color in palette]
    cases_colors = {case_id: palette[i] for i, case_id in enumerate(set(schedule['room_id']))}

    for i, (resource_on_block_id, resource, evt) in enumerate(
            zip(schedule['room_id'], schedule['anesthetist_id'], intervals)):
        txt_to_print = re.search(r'\d+', resource_on_block_id).group()
        ax.barh(resource_mapping[resource], width=evt[1] - evt[0], left=evt[0], linewidth=1, edgecolor='black',
                color=cases_colors[resource_on_block_id])
        ax.text((evt[0] + evt[1] - 0.07 * len(str(txt_to_print))) / 2, resource_mapping[resource], txt_to_print,
                fontname='Arial', color='white', va='center')

    # Enable zooming with mplcursors
    mplcursors.cursor(hover=True)

    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels([f'{resource}' for resource in resources])

    ax.set_ylabel('anesthetist_id'.replace('_', ' '))
    ax.set_title(f'Total {len(set(schedule["anesthetist_id"]))} anesthetists')

# cost calculation
def calculate_cost(duration):
    regular_rate = max(5, duration)
    overtime_rate = 0.5 * max(0, duration - 9)
    return regular_rate + overtime_rate


def get_last_date_and_new_id(anesthetists):
    if not anesthetists:
        # If anesthetists is empty, return a default start date and the first ID
        start_date = pd.to_datetime('1900-01-01')
        new_anesthetist_id = 'anesthetist-0'
    else:
        # Extract the last date from the existing anesthetists
        start_date = max(anesthetists.values(), key=lambda x: x['shift_end'])['shift_end']

        # Extract the last ID and create a new one with the next ID
        last_id = max(int(anesthetist_id.split('-')[-1]) for anesthetist_id in anesthetists.keys())
        new_id = last_id + 1
        new_anesthetist_id = f'anesthetist-{new_id}'

    return start_date, new_anesthetist_id


#plot rectangles rooms
def plot_patches(df):
     # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot rectangles for each row in the DataFrame
    for index, row in df.iterrows():
        rect = patches.Rectangle(
            (row['intervals_start'], df['room'].unique().tolist().index(row['room'])),
            row['intervals_end'] - row['intervals_start'],
            0.8,  # Height of the rectangle (adjust as needed)
            linewidth=4,
            edgecolor='black',
            facecolor='none' if row['duration'] == 0 else 'lightblue',  # Use 'lightblue' color for non-zero duration
        )
        ax.add_patch(rect)

    # Set y-axis ticks and labels
    ax.set_yticks(range(len(df['room'].unique())))
    ax.set_yticklabels(df['room'].unique())

    # Set x-axis label
    ax.set_xlabel('Time')

# assign rooms by schedule
def getScheduleRooms(rooms,dfbig):
    schedule = []

    for _, surgery in dfbig.iterrows():
        for room_id, room in rooms.items():

            #free room
            if(room['intervals_end'] <= surgery.intervals_start):
                room['available'] = True
                room['intervals_end'] = 0.0

            if(room['available']):
                schedule.append(room_id)

                room['available'] = False
                room['intervals_end'] = surgery.intervals_end

                duration = surgery.duration


                break

    return   schedule

def initrooms():
    rooms = {f'room-{i}': {'available': True, 'intervals_end': 0.0} for i in range(20)}
    return rooms

# schedule every surgery new doctor
def getScheduleRoomsAndDoc(rooms,dfbig):
    schedule = []
    doctor   = []
    cost     =  []
    id = 0
    total_cost = 0
    for _, surgery in dfbig.iterrows():

        for room_id, room in rooms.items():

            #free room
            if(room['intervals_end'] <= surgery.intervals_start):
                room['available'] = True
                room['intervals_end'] = 0.0

            if(room['available']):
                schedule.append(room_id)
                doctor.append(f'anesthetist - {id}')
                id=id+1

                room['available'] = False
                room['intervals_end'] = surgery.intervals_end

                duration = surgery.duration
                total_cost += calculate_cost(duration)
                cost.append(total_cost)
                break

    return   total_cost,schedule,doctor,cost

# schedule  surgery : room and doctor: , we look 1 forward if we can use current or new doctor
def getScheduleRoomsAndDocDepth1(rooms,dfbig):
    schedule = []
    doctor   = []
    cost     =  [] # cost sum added
    cost_row     =  [] # cost curent surgery

    anesthetists ={}

    id = 0
    total_cost = 0
    for _, surgery in dfbig.iterrows():

        #if surgery['Unnamed: 0']==47:
        #  print("Column 'Unnamed: 0' is present.")

        for room_id, room in rooms.items():

            #free room
            if(room['available']==False):
                if(room['intervals_end'] <= surgery.intervals_start):
                    room['available'] = True
                    room['intervals_end'] = 0.0

            else:
                schedule.append(room_id)
                room['available'] = False
                room['intervals_end'] = surgery.intervals_end

                #find if we have free doctor +less price else we add new one
                cur_id=0
                selected_id=-1
                if len(anesthetists) == 0:
                    # Adding a new anesthetist
                    new_anesthetist_id = id  # Assuming you want the ID to be 0 for the new anesthetist
                    new_anesthetist_key = f'anesthetist-{new_anesthetist_id}'

                    cur_id = id
                    duration = surgery.duration
                    cost_cur = calculate_cost(duration)

                    #new
                    anesthetists[new_anesthetist_key] = {
                        'shift_start': surgery.start,
                        'shift_end': surgery.end,
                        'busy': True,
                        'duration':surgery.duration,
                        'cost': cost_cur}

                else:
                    #1 reset doctor states
                    for anesthetist_id, anesthetist in anesthetists.items():
                        if anesthetist['shift_end']<=surgery.start:
                            anesthetist['busy']=False

                    duration = surgery.duration
                    cost_cur = calculate_cost(duration)

                    # run over doctor and see if we have better cost to reuse
                    selected_id = -1
                    for anesthetist_id, anesthetist in anesthetists.items():
                        if anesthetist['busy']==False:
                           duration_anth  =  (surgery.end-anesthetist['shift_start']).total_seconds() / 3600

                           #  NOT MORE THAN 12
                           if(duration_anth > MAX_TIME):
                               continue

                           cost_anth      =  calculate_cost(duration_anth)

                           # we must select to achive min time !
                           if(anesthetist['duration']< MIN_TIME):
                                    selected_id = anesthetist_id
                                    cost_cur = cost_anth
                                    cur_id = anesthetist_id
                                    break

                           # doctor is better or same than add new
                           # if selected doctor first we remain with choice
                           # DOCTOR SHOULD WORK AT LEAST 5 HOURS BUT
                           if((selected_id==-1 and cost_anth<=cost_cur) or (cost_anth<cost_cur)) :
                                       selected_id = anesthetist_id
                                       cost_cur = cost_anth
                                       cur_id = anesthetist_id


                    #id=cur_id , found better solution than add new
                    if(selected_id!=-1):
                           start = anesthetists[selected_id]['shift_start']
                           durationTotal  = (surgery.end - start).total_seconds() / 3600
                           anesthetists[selected_id] = {
                            'shift_start': start, #remain
                            'shift_end': surgery.end,
                            'busy': True,
                            'duration':durationTotal,
                            'cost':cost_cur}


                    else:
                      #add new doctor
                      new_anesthetist_id =  len(anesthetists)
                      cur_id = new_anesthetist_id
                      new_anesthetist_key = f'anesthetist-{new_anesthetist_id}'
                      duration = surgery.duration
                      cost_cur = calculate_cost(duration)

                      anesthetists[new_anesthetist_key] = {
                            'shift_start': surgery.start, # new doctor
                            'shift_end': surgery.end,
                            'busy': True,
                            'duration':surgery.duration,
                            'cost':cost_cur
                            }

                if(selected_id!=-1):
                    cost_cur -= anesthetists[selected_id]['cost'] # update to total cost


                cost_row.append(cost_cur)
                total_cost +=cost_cur #calculate_cost(duration)
                cost.append(total_cost)

                if(selected_id!=-1):
                    doctor.append(selected_id)
                else:
                   #new doctor id
                   doctor.append(f'anesthetist-{cur_id}')
                break

    pprint(anesthetists)
    # Convert the dictionary to a DataFrame
    anesthetists_df = pd.DataFrame.from_dict(anesthetists, orient='index')

    # Write the DataFrame to a CSV file
    anesthetists_df.to_csv('anesthetists_dataD1.csv', index_label='Anesthetist_ID')


    return   total_cost,schedule,doctor,cost

#plot rooms schedule
def plot_schedule(rooms_plot_numeric, interval_start_end):
    fig, ax = plt.subplots()

    unique_elements = list(set(rooms_plot_numeric))
    color_mapping = {room: plt.cm.get_cmap('tab20')(i) for i, room in enumerate(unique_elements)}

    for room, interval in zip(rooms_plot_numeric, interval_start_end):
        #print(room,interval)
        room_color = color_mapping[room]
        rect = patches.Rectangle((room, interval[0]), 1, interval[1] - interval[0], linewidth=1,
                                 edgecolor='black', facecolor=room_color)
        ax.add_patch(rect)
        #print('.')




    start_times = [interval[0] for interval in interval_start_end]
    end_times = [interval[1] for interval in interval_start_end]

    # Find the minimum and maximum values
    min_start_time = min(start_times)
    max_end_time = max(end_times)

    ax.set_xlim(-1, max(rooms_plot_numeric) + 2)
    #ax.set_ylim(6, 8)  # Adjust the y-axis limits as needed
    ax.set_ylim(np.floor(min_start_time-1),np.ceil(max_end_time+1))
    ax.set_xlabel('Rooms')
    ax.set_ylabel('Time')

def createIntervals(dfbig):

    # Create a list of rectangle interval start and end
    interval_start_end = []

    # Iterate over the dataframe
    for index, row in dfbig.iterrows():
        # Get the interval start and end values
        interval_start = row["intervals_start"]
        interval_end = row["intervals_end"]

        # Add the interval start and end to the list
        interval_start_end.append([interval_start, interval_end])

    return     interval_start_end

dfSample = pd.read_csv('example_sol.csv')
#print(dfSample)

dfSample['start_time'] = pd.to_datetime(dfSample['start_time'], format='%d/%m/%Y %H:%M')
dfSample['end_time'] = pd.to_datetime(dfSample['end_time'], format='%d/%m/%Y %H:%M')

#plot_day_schedule(df)
#plot_day_scheduleDay(dfSample)

x=[]
y=[]
for d in range(0,13):
    #print(' Hours ',d,' Payment ',payment(d))
    x.append(d)
    y.append(calculate_cost(d))

plt.figure()
plt.plot(x,y)
plt.xlabel('duration H')
plt.ylabel('Payment')
plt.grid(True)
plt.title('Payment vs Duration[H]')

#-------------------
'''
a greedy algorithm approach. 
The idea is to iteratively assign surgeries to anesthesiologists and rooms
while minimizing the total cost
'''
#-------------------

dfbig = pd.read_csv("surgeries.csv")
print(dfbig.head())

# Convert start_time and end_time to datetime objects
dfbig['start'] = pd.to_datetime(dfbig['start'],format='%d/%m/%Y %H:%M')
dfbig['end'] = pd.to_datetime(dfbig['end'],format='%d/%m/%Y %H:%M')

# Calculate time intervals in hours from the start of the day
intervals_start = (dfbig['start'] - dfbig['start'].dt.floor('d')).dt.total_seconds().div(3600)
intervals_end = (dfbig['end'] - dfbig['start'].dt.floor('d')).dt.total_seconds().div(3600)

dfbig['intervals_start']=intervals_start
dfbig['intervals_end']=intervals_end

# Calculate duration in hours
dfbig['duration'] = intervals_end - intervals_start

# If you want to round the duration to a certain number of decimal places (e.g., 2 decimal places)
dfbig['duration'] = dfbig['duration'].round(2)


#dfbig.to_csv('dfbig.csv', index=False)

print(dfbig.head())



#===============================


#rooms
rooms = initrooms()
dfbig['room'] = getScheduleRooms(rooms,dfbig)



#every procedure new doctor
rooms = initrooms()
total_costSimple,scheduleSimple,doctorSimple,cost = getScheduleRoomsAndDoc(rooms,dfbig)
dfbig['doctorSimple'] = doctorSimple
dfbig['cost'] = cost
#dfbig.to_csv('dfbigRoomDoctor.csv', index=False)
#plot_patches(dfbig)
print('simple cost',total_costSimple)

#---------------
# depth1
#todo: add limit hours
rooms = initrooms()
total_costSimple1,scheduleSimple1,doctorSimple1,cost1 = getScheduleRoomsAndDocDepth1(rooms,dfbig)
print('simple cost1',total_costSimple1)

dfbig['doctorSimple'] = doctorSimple1
dfbig['cost'] = cost1
#-----------------
#map to plot
# Create a new DataFrame with modified column names
new_columns_mapping = {
    'Unnamed: 0':     'Unnamed: 0',
    'start': 'start_time',
    'end': 'end_time',
    'doctorSimple': 'anesthetist_id',
    'room': 'room_id',
}

new_df = dfbig.rename(columns=new_columns_mapping)
new_df = new_df.drop('intervals_start', axis=1)
new_df = new_df.drop('intervals_end', axis=1)
new_df = new_df.drop('duration', axis=1)
new_df = new_df.drop('cost', axis=1)
plot_day_scheduleDay(new_df)
#===============
interval_start_end = createIntervals(dfbig)

# Create a bar chart
roomsPlot = dfbig['room']
room_mapping = {f'room-{i}': i for i in range(20)}
rooms_plot_numeric = list(map(lambda room: room_mapping[room], roomsPlot))
#plt.bar(rooms_plot_numeric, interval_start_end)
'''
fig, ax = plt.subplots()

# Plot rectangles for each data point
for room, (start, end) in zip(rooms_plot_numeric, interval_start_end):
    rect = plt.Rectangle((start, room), (end - start)*10, 1, edgecolor='black', facecolor='lightblue', linewidth=4)
    ax.add_patch(rect)

# Set y-axis ticks and labels
ax.set_yticks(range(len(rooms_plot_numeric)))
ax.set_yticklabels(rooms_plot_numeric)

# Set x-axis label
ax.set_xlabel('Time')

# Show the plot

# Set the y-label
plt.ylabel("Rooms (0-19)")

# Set the x-label
plt.xlabel("Rectangle Interval Start and End")

# Set the title of the plot
plt.title("Room Duration")
'''


#plot_schedule(rooms_plot_numeric, interval_start_end)
plot_schedule(rooms_plot_numeric, interval_start_end)

#dfbig.to_csv('dfbigRoom.csv', index=False)
#dfbig.to_csv('dfbigRoomD1.csv', index=False)
#dfbig.to_csv('dfbigRoomD1C.csv', index=False)

#'start_time', 'end_time', 'anesthetist_id', 'room_id'


plt.show()
print('.')
