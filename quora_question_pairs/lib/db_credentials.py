import MySQLdb

def db_credentials():
    return "[user]", "[pass]", "[db_name]"

def openmysql():
    db = MySQLdb.connect("localhost", "[user]", "[pass]", "[db_name]", use_unicode=True, charset="utf8")
    cursor = db.cursor()
    return db, cursor

def closemysql(db):
    db.close()
