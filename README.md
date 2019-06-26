# sotabench

Easily benchmark PyTorch models on selected tasks and datasets.

## Example Usage

```python
import sotabench.image_classification.cifar10 as cifar10

# model = ... (returns a nn.Module object)

cifar10.benchmark(model=model)
# returns a dictionary with evaluation information
```

## How to run in context of an example repository

Clone the example repository and repo into it. You should see a hubconf.py file; this
will contain an sotabench call (like above) - this is what you need to run to capture
the evaluation output.

## Making sure evaluation creates a json

Set the following environment variable in production:

```bash
export SOTABENCH_STORE_RESULTS=true
```


## Prepopulated Data Cache

When the evaluation script is run, it typically includes a PyTorch Dataset 
class which attempts to download a raw file. These raw files can be quite 
big (GBs). But these classes will check whether files have already been downloaded.
Therefore we can prepopulate the data folder with the raw files, so we don't
have to download every time (which can take a long time).

The standard data root directory I am assuming is `./data` - this will be in the 
same folder where the evaluation script is.

In order to prepopulate, I have made a S3 bucket with lots of raw data files that
are used by PyTorch Dataset classes. Install the following:

```bash
pip install awscli
pip install boto3
```

Then configure with aws configure (get your access keys from AWS console).
Here is a script for getting the data - feel free to put wherever you think is best
on the server. Just bear in mind the evaluation script will look in the same place for
the data folder, so might need to move folder to same place temporarily. 

```
import boto3
import os

BUCKET_NAME = 'sotabench'
OUTPUT_DIR = './data/'

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

s3 = boto3.client('s3')
list = s3.list_objects(Bucket=BUCKET_NAME)['Contents']
files = [i for i in s3.list_objects(Bucket='sotabench')['Contents'] if i['Key'][-1] != '/']

for file in files:
    print(file['Key'])
    output_location = '%s%s' % (OUTPUT_DIR, file['Key'].split('/')[-1])
    s3.download_file(BUCKET_NAME, file['Key'], output_location)
```