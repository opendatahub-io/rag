"""Pydantic models for WhatsApp webhook data."""

from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime


class MessageKey(BaseModel):
    """WhatsApp message key structure."""
    remoteJid: str
    fromMe: bool
    id: str
    participant: Optional[str] = None


class MessageInfo(BaseModel):
    """WhatsApp message info structure."""
    key: MessageKey
    messageTimestamp: Optional[int] = None
    pushName: Optional[str] = None
    message: Optional[Dict[str, Any]] = None


class WebhookData(BaseModel):
    """Webhook data structure from Evolution API."""
    event: str
    instance: str
    data: Dict[str, Any]
    destination: Optional[str] = None
    date_time: Optional[str] = None
    sender: Optional[str] = None
    server_url: Optional[str] = None
    apikey: Optional[str] = None


class GroupInfo(BaseModel):
    """Group information structure."""
    id: str
    subject: str
    subjectOwner: Optional[str] = None
    subjectTime: Optional[int] = None
    creation: Optional[int] = None
    owner: Optional[str] = None
    desc: Optional[str] = None
    descId: Optional[str] = None
    restrict: Optional[bool] = None
    announce: Optional[bool] = None
    participants: Optional[List[Dict[str, Any]]] = None
