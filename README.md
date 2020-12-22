# Deploy Transformers Downstream with FastAPI

Deploy a pre-trained transformers model as a REST API using FastAPI

## Demo

request example
```bash
curl -H "Content-type: application/json" -X POST -d '{"passage":"I are a boy"}' http://localhost:8000/generate 
```

Then you will get reponse like this

```js
{"prob": 0.001}
```

You can also [read the complete tutorial here](https://www.curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)



## License

MIT
