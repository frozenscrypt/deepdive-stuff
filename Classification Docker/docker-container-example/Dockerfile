FROM python:3.8.8

# Step 2: Add requirements.txt file 
COPY ./requirements.txt /requirements.txt

# Step 3:  Install required pyhton dependencies from requirements file
RUN pip install -r requirements.txt


ADD . /app


# Step 5: Set working directory to previously added app directory
WORKDIR /app

COPY . /app

# Step 6: Expose the port Flask is running on
EXPOSE 5000

# ENTRYPOINT ["python"]

# Step 9: Run Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

