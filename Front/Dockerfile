# pull official base image
FROM node:10

# set working directory
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
#copies package.json and package-lock.json to Docker environment
COPY package.json ./
COPY package-lock.json ./
# Installs all node packages
RUN npm install 


# Copies everything over to Docker environment
COPY . ./
EXPOSE 3000
# start app
CMD ["npm", "start"]