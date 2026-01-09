from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from app.services.parsing import extract_text_from_file, parse_with_openai
from app.schema import ExtractResponse, Invoice

app = FastAPI(title="Invoice Parsing API")

@app.post("/extract", response_model=ExtractResponse, tags=["Upload"])
async def extract_invoice(
    userID: str = Form(...),
    file: UploadFile = File(...)
):
    raw_text = await extract_text_from_file(file)

    invoice_data = await parse_with_openai(raw_text)

    invoice = Invoice(**invoice_data)

    return ExtractResponse(
        userID=userID,
        invoice=invoice
    )

@app.get("/")
async def root():
    return {"message": "Invoice Parsing API is running."}