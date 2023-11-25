use postgres::Client;

pub fn import_index(client: &Client, index_data: &[u8]) {
    client.batch_execute(&format!(
        "
CREATE OR REPLACE FUNCTION import_index_file() RETURNS VOID AS $$
DECLARE
  lo_oid OID;
  fd INT;
BEGIN
    lo_oid := pg_catalog.lo_create(0);
    fd := pg_catalog.lo_open(lo_oid, 131072);
    PERFORM pg_catalog.lowrite(fd, 'test1');
    PERFORM pg_catalog.lowrite(fd, 'test2');
    PERFORM lo_export(lo_oid, '/tmp/index.usearch');
    -- then change in lantern to read from lo instead of file
END;
$$ LANGUAGE plpgsql;
",
        index_data
    ));
}
