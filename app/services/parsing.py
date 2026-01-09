import asyncio
from openai import OpenAI
import json
import os
from langdetect import detect
from googletrans import Translator

def detect_and_translate_to_english(text):
    detected_lang = detect(text)
    if detected_lang == 'en':
        return text
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

from app.config import OPENAI_API_KEY, OPENAI_MODEL

async def extract_text_from_file(file) -> str:
    def _sync_extract():
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set the OPENAI_API_KEY environment variable or configure it in app/config.py"
            )

        filename = (getattr(file, "filename", None) or "upload").lower()

        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            file.file.seek(0)
        except Exception:
            pass

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

    return await asyncio.to_thread(_sync_extract)


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
            "Do not guess values. "
            "For the 'type' field, only use 'CLIENT' or 'COMPANY' (or null if not present)."
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
                            "companyName",
                            "email",
                            "AddressAndContactInfo",
                            "projectInformation",
                            "projectDescription",
                            "serviceAndItems",
                            "vat",
                            "subTotal",
                            "totalAmount",
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
                                            "total",
                                        ],
                                        "properties": {
                                            "name": {"type": ["string", "null"]},
                                            "quantity": {"type": ["number", "null"]},
                                            "unitPrice": {"type": ["number", "null"]},
                                            "total": {"type": ["number", "null"]}
                                        }
                                    }
                                },

                                "vat": {"type": ["number", "null"]},
                                "subTotal": {"type": ["number", "null"]},
                                "totalAmount": {"type": ["number", "null"]},

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