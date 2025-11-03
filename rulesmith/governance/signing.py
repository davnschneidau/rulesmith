"""Rulebook signing and attestation."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from rulesmith.io.ser import RulebookSpec


@dataclass
class Signature:
    """Digital signature for a rulebook."""
    
    signer: str
    timestamp: str
    signature_hash: str
    algorithm: str = "SHA256"
    certificate: Optional[str] = None  # Base64-encoded certificate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signer": self.signer,
            "timestamp": self.timestamp,
            "signature_hash": self.signature_hash,
            "algorithm": self.algorithm,
            "certificate": self.certificate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signature":
        """Create from dictionary."""
        return cls(
            signer=data["signer"],
            timestamp=data["timestamp"],
            signature_hash=data["signature_hash"],
            algorithm=data.get("algorithm", "SHA256"),
            certificate=data.get("certificate"),
        )


@dataclass
class Attestation:
    """Attestation for a rulebook (SBOM, provenance, etc.)."""
    
    rulebook_name: str
    rulebook_version: str
    signatures: List[Signature] = field(default_factory=list)
    sbom: Optional[Dict[str, Any]] = None  # Software Bill of Materials
    provenance: Optional[Dict[str, Any]] = None  # Provenance information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rulebook_name": self.rulebook_name,
            "rulebook_version": self.rulebook_version,
            "signatures": [s.to_dict() for s in self.signatures],
            "sbom": self.sbom,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attestation":
        """Create from dictionary."""
        return cls(
            rulebook_name=data["rulebook_name"],
            rulebook_version=data["rulebook_version"],
            signatures=[Signature.from_dict(s) for s in data.get("signatures", [])],
            sbom=data.get("sbom"),
            provenance=data.get("provenance"),
            metadata=data.get("metadata", {}),
        )


class RulebookSigner:
    """Signs and verifies rulebook signatures."""
    
    def __init__(self, signing_key: Optional[str] = None):
        """
        Initialize rulebook signer.
        
        Args:
            signing_key: Optional signing key (for production, use proper key management)
        """
        self.signing_key = signing_key
    
    def compute_hash(self, spec: RulebookSpec) -> str:
        """
        Compute hash of rulebook spec.
        
        Args:
            spec: Rulebook spec
        
        Returns:
            SHA256 hash
        """
        # Serialize spec to deterministic JSON
        spec_dict = {
            "name": spec.name,
            "version": spec.version,
            "nodes": [
                {
                    "name": n.name,
                    "kind": n.kind,
                    "rule_ref": n.rule_ref,
                    "model_uri": n.model_uri,
                }
                for n in spec.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target}
                for e in spec.edges
            ],
        }
        
        spec_json = json.dumps(spec_dict, sort_keys=True)
        return hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
    
    def sign(
        self,
        spec: RulebookSpec,
        signer: str,
        certificate: Optional[str] = None,
    ) -> Signature:
        """
        Sign a rulebook spec.
        
        Args:
            spec: Rulebook spec to sign
            signer: Signer identifier
            certificate: Optional certificate
        
        Returns:
            Signature object
        """
        spec_hash = self.compute_hash(spec)
        
        # In production, this would use proper cryptographic signing
        # For now, we create a hash-based signature
        if self.signing_key:
            # Combine hash with signing key
            combined = f"{spec_hash}:{self.signing_key}"
            signature_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        else:
            # Simple hash-based signature
            signature_hash = hashlib.sha256(f"{spec_hash}:{signer}".encode("utf-8")).hexdigest()
        
        return Signature(
            signer=signer,
            timestamp=datetime.utcnow().isoformat(),
            signature_hash=signature_hash,
            algorithm="SHA256",
            certificate=certificate,
        )
    
    def verify(
        self,
        spec: RulebookSpec,
        signature: Signature,
    ) -> bool:
        """
        Verify a rulebook signature.
        
        Args:
            spec: Rulebook spec
            signature: Signature to verify
        
        Returns:
            True if signature is valid
        """
        expected_hash = self.compute_hash(spec)
        
        if self.signing_key:
            combined = f"{expected_hash}:{self.signing_key}"
            expected_signature = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        else:
            expected_signature = hashlib.sha256(f"{expected_hash}:{signature.signer}".encode("utf-8")).hexdigest()
        
        return signature.signature_hash == expected_signature
    
    def create_attestation(
        self,
        spec: RulebookSpec,
        signer: str,
        sbom: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        certificate: Optional[str] = None,
    ) -> Attestation:
        """
        Create attestation for a rulebook.
        
        Args:
            spec: Rulebook spec
            signer: Signer identifier
            sbom: Optional Software Bill of Materials
            provenance: Optional provenance information
            certificate: Optional certificate
        
        Returns:
            Attestation object
        """
        signature = self.sign(spec, signer, certificate)
        
        # Generate SBOM if not provided
        if sbom is None:
            sbom = self._generate_sbom(spec)
        
        # Generate provenance if not provided
        if provenance is None:
            provenance = self._generate_provenance(spec)
        
        return Attestation(
            rulebook_name=spec.name,
            rulebook_version=spec.version,
            signatures=[signature],
            sbom=sbom,
            provenance=provenance,
        )
    
    def _generate_sbom(self, spec: RulebookSpec) -> Dict[str, Any]:
        """Generate Software Bill of Materials."""
        components = []
        
        for node in spec.nodes:
            component = {
                "name": node.name,
                "type": node.kind,
            }
            
            if node.model_uri:
                component["model_uri"] = node.model_uri
            
            if node.rule_ref:
                component["rule_ref"] = node.rule_ref
            
            components.append(component)
        
        return {
            "format": "SPDX-2.3",
            "components": components,
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def _generate_provenance(self, spec: RulebookSpec) -> Dict[str, Any]:
        """Generate provenance information."""
        return {
            "rulebook_name": spec.name,
            "version": spec.version,
            "node_count": len(spec.nodes),
            "edge_count": len(spec.edges),
            "created_at": datetime.utcnow().isoformat(),
        }


class AttestationStore:
    """Stores and retrieves attestations."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize attestation store.
        
        Args:
            storage_path: Optional path for storing attestations
        """
        self.storage_path = storage_path
        self.attestations: Dict[str, Attestation] = {}  # key: f"{name}@{version}"
    
    def save_attestation(self, attestation: Attestation) -> None:
        """Save an attestation."""
        key = f"{attestation.rulebook_name}@{attestation.rulebook_version}"
        self.attestations[key] = attestation
        
        if self.storage_path:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            file_path = os.path.join(self.storage_path, f"{key}.json")
            with open(file_path, "w") as f:
                json.dump(attestation.to_dict(), f, indent=2)
    
    def load_attestation(
        self,
        rulebook_name: str,
        rulebook_version: str,
    ) -> Optional[Attestation]:
        """Load an attestation."""
        key = f"{rulebook_name}@{rulebook_version}"
        
        if key in self.attestations:
            return self.attestations[key]
        
        if self.storage_path:
            import os
            file_path = os.path.join(self.storage_path, f"{key}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                return Attestation.from_dict(data)
        
        return None


# Global signer and store
rulebook_signer = RulebookSigner()
attestation_store = AttestationStore()

