import pytest
from uuid import uuid4

# Unit tests validating the RLS predicate logic defined in 008_fix_resume_sections_rls.sql
# Predicate:
# auth_uid == section_user_id AND (resume_id, auth_uid) in resumes_table

class MockDatabase:
    def __init__(self):
        # resumes: {resume_id: owner_user_id}
        self.resumes = {}
        # resume_sections: {section_id: {"resume_id": ..., "user_id": ..., "title": ...}}
        self.resume_sections = {}

    def insert_resume(self, resume_id, user_id):
        self.resumes[resume_id] = user_id

    # Simulation of Postgres RLS for resume_sections
    def can_select(self, auth_uid, section_id):
        sec = self.resume_sections.get(section_id)
        if not sec:
            return False
        # auth.uid() = user_id AND EXISTS (SELECT 1 FROM resumes WHERE id = resume_id AND user_id = auth.uid())
        is_user_match = (auth_uid == sec["user_id"])
        is_resume_owner = (self.resumes.get(sec["resume_id"]) == auth_uid)
        return is_user_match and is_resume_owner

    def can_insert(self, auth_uid, section_data):
        # WITH CHECK (auth.uid() = user_id AND EXISTS (SELECT 1 FROM resumes WHERE id = resume_id AND user_id = auth.uid()))
        is_user_match = (auth_uid == section_data["user_id"])
        is_resume_owner = (self.resumes.get(section_data["resume_id"]) == auth_uid)
        return is_user_match and is_resume_owner

    def can_update(self, auth_uid, section_id, new_data):
        sec = self.resume_sections.get(section_id)
        if not sec:
            return False
        # USING clause
        using_ok = (auth_uid == sec["user_id"]) and (self.resumes.get(sec["resume_id"]) == auth_uid)
        # WITH CHECK clause
        check_ok = (auth_uid == new_data.get("user_id", sec["user_id"])) and (self.resumes.get(new_data.get("resume_id", sec["resume_id"])) == auth_uid)
        return using_ok and check_ok

    def can_delete(self, auth_uid, section_id):
        sec = self.resume_sections.get(section_id)
        if not sec:
            return False
        # USING (auth.uid() = user_id AND EXISTS (SELECT 1 FROM resumes WHERE id = resume_id AND user_id = auth.uid()))
        return (auth_uid == sec["user_id"]) and (self.resumes.get(sec["resume_id"]) == auth_uid)


def test_cross_user_isolation_on_resume_sections():
    db = MockDatabase()
    
    user_a = str(uuid4())
    user_b = str(uuid4())
    
    resume_a = str(uuid4())
    resume_b = str(uuid4())
    
    # User A creates Resume A; User B creates Resume B
    db.insert_resume(resume_a, user_a)
    db.insert_resume(resume_b, user_b)
    
    # 1. User A inserts section into Resume A -> ALLOWED
    section_a_id = str(uuid4())
    sec_a_data = {
        "id": section_a_id,
        "resume_id": resume_a,
        "user_id": user_a,
        "title": "User A Experience"
    }
    assert db.can_insert(user_a, sec_a_data) is True
    db.resume_sections[section_a_id] = sec_a_data
    
    # 2. User B tries to insert section into User A's Resume A with User B's user_id -> BLOCKED
    # (Under old migration 006, this succeeded because it only checked auth.uid() = user_id!)
    attack_sec_b = {
        "id": str(uuid4()),
        "resume_id": resume_a,  # Target is User A's resume
        "user_id": user_b,      # User B sets their own user_id
        "title": "Malicious Injection"
    }
    assert db.can_insert(user_b, attack_sec_b) is False, "Cross-user section injection MUST be blocked by RLS"
    
    # 3. User B tries to insert section into User A's Resume A pretending to be User A -> BLOCKED (auth.uid() != user_id)
    spoof_sec = {
        "id": str(uuid4()),
        "resume_id": resume_a,
        "user_id": user_a,
        "title": "Spoofed Injection"
    }
    assert db.can_insert(user_b, spoof_sec) is False, "Spoofed user_id insertion MUST be blocked by RLS"
    
    # 4. User B tries to SELECT User A's section -> BLOCKED
    assert db.can_select(user_b, section_a_id) is False, "User B must not be able to read User A's section"
    assert db.can_select(user_a, section_a_id) is True, "User A must be able to read own section"
    
    # 5. User B tries to UPDATE User A's section -> BLOCKED
    update_data = dict(sec_a_data)
    update_data["title"] = "Hacked Section Title"
    assert db.can_update(user_b, section_a_id, update_data) is False, "User B must not be able to update User A's section"
    assert db.can_update(user_a, section_a_id, update_data) is True, "User A must be able to update own section"
    
    # 6. User B tries to DELETE User A's section -> BLOCKED
    assert db.can_delete(user_b, section_a_id) is False, "User B must not be able to delete User A's section"
    assert db.can_delete(user_a, section_a_id) is True, "User A must be able to delete own section"
