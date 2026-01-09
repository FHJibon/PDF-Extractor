import asyncio
from openai import OpenAI
try:
    from openai import BadRequestError
except Exception: 
    BadRequestError = Exception
import json
import os

from app.config import OPENAI_API_KEY, OPENAI_MODEL


def _chat_completions_create_compat(
    client: OpenAI,
    *,
    model: str,
    messages: list,
    max_output_tokens: int,
    temperature: float = 0.0,
):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_output_tokens,
            temperature=temperature,
        )
    except BadRequestError as exc:
        message = str(getattr(exc, "message", "") or exc)
        if "max_completion_tokens" in message or "Unsupported parameter" in message:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
        raise

async def extract_text_from_file(file) -> str:
    filename = (getattr(file, "filename", None) or "upload").lower()
    ext = os.path.splitext(filename)[1].lower()
    client = OpenAI(api_key=OPENAI_API_KEY)

    if ext in [".jpg", ".jpeg", ".png"]:
        file.file.seek(0)
        image_bytes = file.file.read()
        import base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        vision_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract ALL readable text from this image. Return plain text only. Do not summarize."},
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext[1:]};base64,{image_b64}"}}
                ]
            }
        ]
        response = _chat_completions_create_compat(
            client,
            model=OPENAI_MODEL,
            messages=vision_prompt,
            max_output_tokens=2048,
            temperature=0.0,
        )
        text = response.choices[0].message.content
    elif ext == ".docx":
        from docx import Document
        file.file.seek(0)
        with open("_temp.docx", "wb") as temp_docx:
            temp_docx.write(file.file.read())
        doc = Document("_temp.docx")
        text = "\n".join([para.text for para in doc.paragraphs])
        os.remove("_temp.docx")
    else:
        def _sync_extract():
            if not OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set."
                )
            file.file.seek(0)
            uploaded = client.files.create(
                file=(filename, file.file),
                purpose="assistants",
            )
            response = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Extract ALL readable text from the attached document. "
                                    "Return plain text only. Do not summarize."
                                ),
                            },
                            {"type": "input_file", "file_id": uploaded.id},
                        ],
                    }
                ],
                temperature=0.0,
            )
            return response.output_text
        text = await asyncio.to_thread(_sync_extract)

    from langdetect import detect
    detected_lang = detect(text)
    if detected_lang != 'en':
        translation_prompt = f"Translate the following text to English. Return only the translated plain text.\n\n{text}"
        response = _chat_completions_create_compat(
            client,
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": translation_prompt}],
            max_output_tokens=2048,
            temperature=0.0,
        )
        text = response.choices[0].message.content
    return text

async def parse_with_openai(raw_text: str) -> dict:
    def _normalize_invoice_payload(data: dict) -> dict:
        if not isinstance(data, dict):
            return {}

        normalized = dict(data)

        invoice_type = normalized.get("type")
        if isinstance(invoice_type, str):
            invoice_type = invoice_type.strip().upper()
        if invoice_type not in ("CLIENT", "COMPANY", None):
            normalized["type"] = None
        else:
            normalized["type"] = invoice_type

        service_items = normalized.get("serviceAndItems")
        if service_items is None:
            normalized["serviceAndItems"] = []
        elif not isinstance(service_items, list):
            normalized["serviceAndItems"] = []

        return normalized

    def _sync_call():
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set the OPENAI_API_KEY environment variable or configure it in app/config.py"
            )

        client = OpenAI(api_key=OPENAI_API_KEY)

        instructions = (
            "You are an invoice extraction engine. "
            "Return ONLY valid JSON. "
            "Missing fields must be null. "
            "Do not guess the values. "
            "For any monetary amount field, also extract its currency from the document. "
        )

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=raw_text,
            instructions=instructions,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "invoice",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "invoiceNo",
                            "issueDate",
                            "dueDate",
                            "type",
                            "currency",
                            "companyName",
                            "email",
                            "AddressAndContactInfo",
                            "projectInformation",
                            "projectDescription",
                            "serviceAndItems",
                            "vat",
                            "vatCurrency",
                            "subTotal",
                            "subTotalCurrency",
                            "totalAmount",
                            "totalAmountCurrency",
                            "isPaid",
                            "paidAt",
                            "additionalNote",
                            "haveAttachment",
                            "attachmentUrl",
                        ],
                        "properties": {
                                "invoiceNo": {"type": ["string", "null"]},
                                "issueDate": {"type": ["string", "null"]},
                                "dueDate": {"type": ["string", "null"]},
                                "type": {
                                    "type": ["string", "null"],
                                    "enum": ["CLIENT", "COMPANY", None],
                                },

                                "currency": {"type": ["string", "null"]},

                                "companyName": {"type": ["string", "null"]},
                                "email": {"type": ["string", "null"]},
                                "AddressAndContactInfo": {"type": ["string", "null"]},

                                "projectInformation": {"type": ["string", "null"]},
                                "projectDescription": {"type": ["string", "null"]},

                                "serviceAndItems": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": [
                                            "name",
                                            "quantity",
                                            "unitPrice",
                                            "unitPriceCurrency",
                                            "total",
                                            "totalCurrency",
                                        ],
                                        "properties": {
                                            "name": {"type": ["string", "null"]},
                                            "quantity": {"type": ["number", "null"]},
                                            "unitPrice": {"type": ["number", "null"]},
                                            "unitPriceCurrency": {"type": ["string", "null"]},
                                            "total": {"type": ["number", "null"]}
                                            ,
                                            "totalCurrency": {"type": ["string", "null"]}
                                        }
                                    }
                                },

                                "vat": {"type": ["number", "null"]},
                                "vatCurrency": {"type": ["string", "null"]},
                                "subTotal": {"type": ["number", "null"]},
                                "subTotalCurrency": {"type": ["string", "null"]},
                                "totalAmount": {"type": ["number", "null"]},
                                "totalAmountCurrency": {"type": ["string", "null"]},

                                "isPaid": {"type": ["boolean", "null"]},
                                "paidAt": {"type": ["string", "null"]},

                                "additionalNote": {"type": ["string", "null"]},
                                "haveAttachment": {"type": ["boolean", "null"]},
                                "attachmentUrl": {"type": ["string", "null"]}
                        }
                    }
                }
            },
            temperature=0.0
        )

        return _normalize_invoice_payload(json.loads(response.output_text))

    return await asyncio.to_thread(_sync_call)