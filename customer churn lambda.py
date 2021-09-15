
import os
import boto3
import codecs
import csv
from io import StringIO
import json

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')


def lambda_handler(event, context):

    #print('event>>>>>>>>',event)

    bucketname = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    client = boto3.client("s3")
    def read_csv_from_s3(bucket_name, key):
        data = client.get_object(Bucket=bucket_name, Key=key)
        # csv_file = io.StringIO()
        # data.to_csv(csv_file, sep=",", header=False, index=False)
        # result = csv_file.getvalue()
        result = []
        for row in csv.DictReader(codecs.getreader("utf-8")(data["Body"])):
            # print(row)
            result.append(row)
            
        return result
    
    result = read_csv_from_s3(bucketname, key)
    
    data = []
    for i in result:
        data_point = []
        for key, value in i.items():
            data_point.append(value)
        data.append(data_point)
    
    
    string_data = ''
    for i in range(len(data)):
        string_data += ",".join(data[i])
        if i==(len(data)-1):
            pass
        else:
            string_data += "\n"
    input_data=tuple(string_data.split("\n"))
    #print(input_data)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=string_data)
    prediction = response['Body'].read().decode()
    #print(prediction)
    
    output=tuple(prediction.split("|"))
    
    result=dict(zip(input_data,output))
    output_data_to_s3=json.dumps(result)
    
    
    upload_path = "mlchurn/Input/outputdata/input_output.txt"
    client.put_object(Body = output_data_to_s3, Bucket=bucketname, Key=upload_path, ContentType='application/json')
    #print('output data stored into s3')
    
    
    
    
    return event
