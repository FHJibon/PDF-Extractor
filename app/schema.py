from typing import Optional, List
from pydantic import BaseModel
from enum import Enum

class InvoiceClientType(str, Enum):
    CLIENT = "CLIENT"
    COMPANY = "COMPANY"

class ServiceAndItem(BaseModel):
    name: Optional[str] = None
    quantity: Optional[float] = None
    unitPrice: Optional[float] = None
    total: Optional[float] = None


class Invoice(BaseModel):
    invoiceNo: Optional[str] = None
    issueDate: Optional[str] = None
    dueDate: Optional[str] = None
    type: Optional[InvoiceClientType] = None

    companyName: Optional[str] = None
    email: Optional[str] = None
    AddressAndContactInfo: Optional[str] = None

    projectInformation: Optional[str] = None
    projectDescription: Optional[str] = None

    serviceAndItems: Optional[List[ServiceAndItem]] = None

    vat: Optional[float] = None
    subTotal: Optional[float] = None
    totalAmount: Optional[float] = None

    isPaid: Optional[bool] = None
    paidAt: Optional[str] = None

    additionalNote: Optional[str] = None

    haveAttachment: Optional[bool] = None
    attachmentUrl: Optional[str] = None

class ExtractResponse(BaseModel):
    userID: str
    invoice: Invoice