VERSION=v11
DOCKERUSER=arpitaambavane89

build:
	docker build -f Dockerfile-sentiment-worker -t sentiment-worker .
push:
	docker tag expense-worker $(DOCKERUSER)/expense-worker:$(VERSION)
	docker push $(DOCKERUSER)/expense-worker:$(VERSION)
	docker tag expense-worker $(DOCKERUSER)/expense-worker:latest
	docker push $(DOCKERUSER)/expense-worker:latest


# ##
# ## The dirkcgrunwald/flair:latest container simply includes the FLAIR library
# ##

# DOCKERUSER_FLAIR=dirkcgrunwald
# build-base:
# 	docker build -f Dockerfile-flair -t flair .
# push-base:
# 	docker tag flair $(DOCKERUSER_FLAIR)/flair:latest
# 	docker push $(DOCKERUSER_FLAIR)/flair:latest