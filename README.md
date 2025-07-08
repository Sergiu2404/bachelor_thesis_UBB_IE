# bachelor_thesis_UBB_IE

Start all the 4 local APIs
./practical_work/server/app_backend_sql
run it as: uvicorn main:app --host 0.0.0.0 --port 8244
./practical_work/server/fin_tinybert_local_api
run it as: uvicorn main --host 0.0.0.0 --port 8771
./practical_work/server/tinybert_credibility_analyzer_local_api
run it as: uvicorn main --host 0.0.0.0 --port 8772
./practical_work/server/price_predictor_local_api
run it as: uvicorn main --host 0.0.0.0 --port 8773

Open ./practical_work/fastapi_auth in android studio, in project: lib/data/services/, IPs inside each of the API files should be the ones of the phone to be connected
or the emmulator's: 10.0.2.2; phone must be connected to same network as host 

