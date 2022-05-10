VERSION=v4
DOCKERUSER=arpitaambavane89

build:
	docker build -f Dockerfile-rest -t expense-rest .

push:
	docker tag sentiment-rest $(DOCKERUSER)/expense-rest:$(VERSION)
	docker push $(DOCKERUSER)/expense-rest:$(VERSION)
	docker tag sentiment-rest $(DOCKERUSER)/expense-rest:latest
	docker push $(DOCKERUSER)/expense-rest:latest
