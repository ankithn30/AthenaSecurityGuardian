import os
import sys
from datetime import datetime
from rapidfuzz import fuzz

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class KnowledgeBase:
    def __init__(self):
        self.room_locations = {
            # Meeting Rooms
            "203": {
                "description": "Room 203 is on the second floor, east wing, next to the elevator",
                "type": "meeting",
                "capacity": "Seats 8-10 people",
                "equipment": "Projector, whiteboard, video conferencing"
            },
            "101": {
                "description": "Room 101 is on the first floor, main entrance area",
                "type": "reception",
                "details": "Main reception and visitor check-in"
            },
            "205": {
                "description": "Room 205 is on the second floor, west wing, near the conference room",
                "type": "meeting",
                "capacity": "Seats 12-15 people",
                "equipment": "Smart TV, conference phone, whiteboard"
            },
            
            # Common Areas
            "bathroom": {
                "description": "Restrooms are located on each floor near the elevator",
                "details": "All restrooms are wheelchair accessible",
                "locations": ["1st floor", "2nd floor", "3rd floor"]
            },
            "cafeteria": {
                "description": "The cafeteria is on the first floor, west wing",
                "hours": "7:30 AM - 3:00 PM, Monday-Friday",
                "payment": "Accepts cash and credit cards",
                "features": "Microwave, vending machines, seating for 50"
            },
            "parking": {
                "description": "Visitor parking is available in the north lot",
                "type": "Outdoor lot",
                "capacity": "50 spaces including 4 handicap spots",
                "payment": "First 2 hours free, $5/hour after"
            },
            "elevator": {
                "description": "Elevators are located in the center of each floor",
                "count": "2 passenger elevators, 1 service elevator",
                "features": "Card access required after hours"
            },
            "exit": {
                "description": "Emergency exits are marked with red signs throughout the building",
                "locations": ["End of each hallway", "Near stairwells", "Main entrance"],
                "type": "Illuminated signs with backup power"
            }
        }

        self.facility_info = {
            "hours": {
                "building": "Building hours are Monday-Friday 7 AM to 8 PM, weekends 9 AM to 5 PM",
                "reception": "Reception is staffed Monday-Friday 8 AM to 6 PM",
                "cafeteria": "Cafeteria serves breakfast and lunch Monday-Friday 7:30 AM to 3 PM",
                "after_hours": "Card access required outside normal hours"
            },
            "security": {
                "desk": "Security desk is staffed 24/7",
                "emergency": "Call extension 911 for emergencies",
                "non_emergency": "For non-emergencies, call extension 555",
                "patrols": "Regular security patrols conducted throughout the building"
            },
            "wifi": {
                "guest": "Guest WiFi network is 'GuestAccess', password is 'visitor123'",
                "coverage": "Available throughout the building",
                "support": "For WiFi issues, contact IT at extension 444",
                "policy": "Acceptable use policy applies to all network users"
            },
            "parking": {
                "visitor": "Visitor parking is free for first 2 hours, $5 per additional hour",
                "location": "North lot - follow signs from main entrance",
                "validation": "Get parking validated at reception",
                "handicap": "Accessible parking available near all entrances"
            },
            "visitor_policy": {
                "check_in": "All visitors must check in at reception and obtain a visitor badge",
                "id": "Government-issued photo ID required",
                "escort": "Visitors must be escorted by an employee at all times",
                "hours": "Visitor check-in available during reception hours only"
            },
        }

        self.emergency_procedures = {
            "general": "Emergency services have been notified. Please remain calm and follow security instructions.",
            "medical": "Medical assistance is on the way. Please do not move unless instructed.",
            "fire": "Fire department notified. Please proceed to nearest exit calmly.",
            "security": "Security team has been alerted and is responding immediately.",
        }

        # synonyms to improve matching
        self.synonyms = {
            "hours": ["hours", "time", "schedule", "open", "close"],
            "wifi": ["wifi", "wi-fi", "internet", "network"],
            "parking": ["parking", "car park", "garage", "lot"],
            "bathroom": ["bathroom", "restroom", "toilet", "washroom"],
            "visitor_policy": ["visitor policy", "guest rules", "check-in policy"],
        }


class ResponseGenerator:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        # Dynamic confidence thresholds based on intent type
        self.confidence_thresholds = {
            "emergency": 0.3,  # Lower threshold for emergency situations
            "greeting": 0.25,  # Very lenient for greetings
            "direction_query": 0.35,  # Balance between accuracy and usability
            "facility_info": 0.35,  # Common queries can be more lenient
            "access_request": 0.4,  # Higher threshold for security
            "visitor_inquiry": 0.35,  # Balance for visitor needs
            "unknown": 0.0,  # Always handle unknown intents
        }
        self.default_threshold = 0.4
        self.context = None  # Store conversation context
        
    def _get_threshold(self, intent):
        """Get confidence threshold for a specific intent"""
        return self.confidence_thresholds.get(intent, self.default_threshold)

    # -------------------------------
    # Public entrypoint
    # -------------------------------
    def generate_response(self, intent_result, original_text):
        """Generate response based on intent classification + text"""
        intent = intent_result.get("intent", "unknown")
        confidence = intent_result.get("confidence", 0.0)
        
        # Store previous context if useful
        self._update_context(intent, original_text)

        # Check confidence against intent-specific threshold
        threshold = self._get_threshold(intent)
        if confidence < threshold:
            # Check if we can use context to improve response
            if self.context and self.context["intent"] == intent:
                # Boost confidence for consistent intent
                confidence += 0.1
            
            if confidence < threshold:
                return self._handle_low_confidence(original_text)

        # Intent handlers with improved context awareness
        response_handlers = {
            "greeting": self._handle_greeting,
            "direction_query": self._handle_directions,
            "access_request": self._handle_access,
            "emergency": self._handle_emergency,
            "visitor_inquiry": self._handle_visitor,
            "facility_info": self._handle_facility_info,
            "unknown": self._handle_unknown,
        }

        handler = response_handlers.get(intent, self._handle_unknown)
        return handler(original_text, confidence)

    # -------------------------------
    # Handlers
    # -------------------------------
    def _handle_greeting(self, text, confidence):
        greetings = [
            "Hello! I'm Athena, your security assistant. How can I help you today?",
            "Good day! I'm here to assist with directions, access, and facility information.",
            "Welcome! How may I assist you with your visit today?",
        ]
        return greetings[hash(text) % len(greetings)]

    def _handle_directions(self, text, confidence):
        text_lower = text.lower()
        found_locations = []
        details_requested = any(word in text_lower for word in ["what", "tell", "more", "detail", "inside", "equipment"])

        # Match against known locations
        for location, info in self.knowledge_base.room_locations.items():
            if self._fuzzy_match(text_lower, location):
                if details_requested:
                    # Provide detailed information
                    response = [info["description"]]
                    if "type" in info:
                        response.append(f"This is a {info['type']} room.")
                    if "capacity" in info:
                        response.append(info["capacity"])
                    if "equipment" in info:
                        response.append(f"Available equipment: {info['equipment']}")
                    found_locations.append(" ".join(response))
                else:
                    found_locations.append(info["description"])

        if found_locations:
            response = found_locations[0]
            if not details_requested and "type" in info:
                response += f" Would you like to know more about this {info['type']} room?"
            return response

        return (
            "I'd be happy to help with directions. "
            "Could you please specify the room number or location you're looking for? "
            "You can ask about meeting rooms, bathrooms, cafeteria, or other facilities."
        )

    def _handle_access(self, text, confidence):
        after_hours = any(phrase in text.lower() for phrase in ["after", "late", "night", "weekend"])
        visitor = any(phrase in text.lower() for phrase in ["visitor", "guest", "visiting"])
        
        if after_hours:
            return (
                f"{self.knowledge_base.facility_info['hours']['after_hours']}. "
                f"Security is available 24/7 at extension 555 if you need assistance."
            )
        elif visitor:
            policy = self.knowledge_base.facility_info['visitor_policy']
            return (
                f"{policy['check_in']} {policy['id']} "
                f"{policy['hours']}"
            )
        else:
            return (
                "For building access, please present your ID at reception. "
                "If you're a visitor, you'll need to check in and obtain a visitor badge. "
                "Would you like to know about visitor policies or after-hours access?"
            )

    def _handle_emergency(self, text, confidence):
        text_lower = text.lower()
        if self._fuzzy_match(text_lower, "medical"):
            return self.knowledge_base.emergency_procedures["medical"]
        elif self._fuzzy_match(text_lower, "fire"):
            return self.knowledge_base.emergency_procedures["fire"]
        elif self._fuzzy_match(text_lower, "security"):
            security_info = self.knowledge_base.facility_info["security"]
            return (
                f"{self.knowledge_base.emergency_procedures['security']} "
                f"{security_info['desk']} {security_info['emergency']}"
            )
        else:
            return (
                f"{self.knowledge_base.emergency_procedures['general']} "
                f"For emergencies, {self.knowledge_base.facility_info['security']['emergency']}"
            )

    def _handle_visitor(self, text, confidence):
        policy = self.knowledge_base.facility_info['visitor_policy']
        parking = self.knowledge_base.facility_info['parking']
        
        if "parking" in text.lower():
            return (
                f"{parking['visitor']} {parking['location']} "
                f"{parking['validation']}"
            )
        elif "id" in text.lower() or "identification" in text.lower():
            return (
                f"{policy['id']} {policy['check_in']}"
            )
        else:
            return (
                f"{policy['check_in']} {policy['id']} "
                f"{policy['escort']} {policy['hours']} "
                "Would you like information about visitor parking?"
            )

    def _handle_facility_info(self, text, confidence):
        text_lower = text.lower()

        # Check for specific subtopics
        for key, info in self.knowledge_base.facility_info.items():
            if isinstance(info, dict):  # Handle nested information
                for subkey, variants in self.knowledge_base.synonyms.items():
                    if any(self._fuzzy_match(text_lower, v) for v in variants):
                        if subkey in info:
                            response = []
                            # Start with main information
                            if "general" in info:
                                response.append(info["general"])
                            # Add specifically requested information
                            response.append(info[subkey])
                            # Add relevant follow-up information
                            if "support" in info and subkey != "support":
                                response.append(info["support"])
                            return " ".join(response)

        # Default fallback
        return (
            "I can provide info on building hours, WiFi, parking, security, and visitor policy. "
            "Which would you like to know about?"
        )

    def _handle_unknown(self, text, confidence):
        return (
            "I can help with directions, access requests, facility information, "
            "visitor check-ins, and emergency situations. What would you like to know?"
        )

    def _handle_low_confidence(self, original_text):
        """Enhanced low confidence handling with context awareness"""
        if not self.context:
            return self._generate_generic_help()
            
        # Try to use context to guide the user
        prev_intent = self.context.get("intent")
        if prev_intent:
            helps = {
                "direction_query": "I heard something about directions. Could you specify which room or area you're looking for?",
                "facility_info": "Were you asking about our facilities? I can tell you about hours, WiFi, parking, or other services.",
                "visitor_inquiry": "If you're asking about visiting, I can help with check-in procedures or visitor policies.",
                "emergency": "Is this an emergency situation? Please clearly state if you need medical, security, or fire assistance.",
            }
            return helps.get(prev_intent, self._generate_generic_help())
        return self._generate_generic_help()

    def _generate_generic_help(self):
        """Generate a context-aware help message"""
        return (
            "I'm not sure I understand. Could you please rephrase that? "
            "I can help with:\n"
            "- Directions to rooms and facilities\n"
            "- Building hours and access\n"
            "- Visitor information\n"
            "- WiFi and parking details\n"
            "- Emergency assistance"
        )

    def _update_context(self, intent, text):
        """Update conversation context"""
        self.context = {
            "intent": intent,
            "text": text,
            "timestamp": datetime.now(),
            "requires_followup": False
        }
        
        # Mark contexts that might need followup
        if intent in ["direction_query", "facility_info"]:
            if "?" in text or "where" in text.lower() or "how" in text.lower():
                self.context["requires_followup"] = True

    # -------------------------------
    # Helpers
    # -------------------------------
    def _fuzzy_match(self, query: str, keyword: str, threshold: int = 75) -> bool:
        """Return True if query approximately matches keyword"""
        # Try direct matching first
        if keyword.lower() in query.lower():
            return True
            
        # Fall back to fuzzy matching
        score = fuzz.partial_ratio(query, keyword)
        
        # Boost score for shorter queries (more likely to be specific)
        if len(query.split()) <= 3:
            score += 10
            
        return score >= threshold


# -------------------------------
# Manual test
# -------------------------------
if __name__ == "__main__":
    generator = ResponseGenerator()
    test_cases = [
        ({"intent": "greeting", "confidence": 0.95}, "Hello there"),
        ({"intent": "direction_query", "confidence": 0.89}, "Where is room 203?"),
        ({"intent": "emergency", "confidence": 0.92}, "Medical emergency"),
        ({"intent": "facility_info", "confidence": 0.87}, "When do you close?"),
        ({"intent": "unknown", "confidence": 0.25}, "Blah blah"),
    ]

    for intent_result, text in test_cases:
        response = generator.generate_response(intent_result, text)
        print(f"Input: {text}")
        print(f"Intent: {intent_result['intent']} ({intent_result['confidence']:.2f})")
        print(f"Response: {response}")
        print("-" * 40)
